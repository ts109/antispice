"""Tests for circuit interpretation and Radau-IIA solver generation."""

import unittest

import wrenfold

import antispice
from antispice import compiler


def make_rc_step_circuit() -> antispice.Circuit:
    """Construct a one-millisecond RC step-response circuit."""
    return antispice.Circuit(
        elements={
            "V1": antispice.Element(
                use="voltage-source",
                nodes=("0", "input"),
                parameters={"voltage": "where(t > 0, 1, 0)"},
            ),
            "R1": antispice.Element(
                use="resistor",
                nodes=("input", "output"),
                parameters={"resistance": 1_000.0},
            ),
            "C1": antispice.Element(
                use="capacitor",
                nodes=("0", "output"),
                parameters={"capacitance": 1e-6},
            ),
        }
    )


class CircuitCompilerTest(unittest.TestCase):
    def test_auxiliary_symbols_are_resolved_and_eliminated(self) -> None:
        model = antispice.Model(
            ports=("ref", "p"),
            parameters=("gain",),
            equations=("I_p - output",),
            auxiliaries={
                "output": "gain * squared",
                "squared": "U_p ** 2",
            },
        )
        system = compiler.compile_circuit(antispice.Circuit(elements={"X1": antispice.Element(model, ("0", "n"), {"gain": 3})}))

        equation = str(system.equations[-1])
        self.assertNotIn("output", equation)
        self.assertNotIn("squared", equation)
        self.assertIn("Phi_n**2", equation)
        self.assertEqual(len(system.state), 2)
        self.assertEqual(tuple(system.auxiliaries), (("X1", "squared"), ("X1", "output")))
        self.assertNotIn("squared", str(system.auxiliaries["X1", "output"]))

    def test_invalid_auxiliary_symbols_are_rejected(self) -> None:
        for auxiliaries, message in (
            ({"a": "b", "b": "a"}, "unresolved or cyclic"),
            ({"U_p": "2 * U_p"}, "collide"),
        ):
            with self.subTest(auxiliaries=auxiliaries):
                model = antispice.Model(
                    ports=("ref", "p"),
                    parameters=(),
                    equations=("I_p",),
                    auxiliaries=auxiliaries,
                )
                circuit = antispice.Circuit(elements={"X1": antispice.Element(model, ("0", "n"))})

                with self.assertRaisesRegex(ValueError, message):
                    compiler.compile_circuit(circuit)

    def test_builtin_transistor_equations_compile_for_both_polarities(self) -> None:
        for polarity in (-1, 1):
            with self.subTest(device="bjt", polarity=polarity):
                system = compiler.compile_circuit(
                    antispice.Circuit(
                        elements={
                            "Q1": antispice.Element(
                                "bjt-ebers-moll",
                                ("0", "base", "collector"),
                                {
                                    "polarity": polarity,
                                    "saturation_current": 1e-15,
                                    "forward_beta": 100,
                                    "reverse_beta": 1,
                                    "thermal_voltage": 0.02585,
                                },
                            )
                        }
                    )
                )
                self.assertEqual(len(system.equations), len(system.state))

            with self.subTest(device="fet", polarity=polarity):
                system = compiler.compile_circuit(
                    antispice.Circuit(
                        elements={
                            "M1": antispice.Element(
                                "fet-shichman-hodges",
                                ("0", "gate", "drain"),
                                {
                                    "polarity": polarity,
                                    "threshold_voltage": 1,
                                    "transconductance": 0.01,
                                    "channel_length_modulation": 0.01,
                                    "transition_voltage": 0.01,
                                },
                            )
                        }
                    )
                )
                self.assertEqual(len(system.equations), len(system.state))

    def test_compiler_creates_deterministic_state_layout(self) -> None:
        system = compiler.compile_circuit(make_rc_step_circuit())

        self.assertEqual(system.layout.potentials, {"input": 0, "output": 1})
        self.assertEqual(
            system.layout.currents,
            {("V1", "p"): 2, ("R1", "p"): 3, ("C1", "p"): 4},
        )
        self.assertEqual(
            tuple(map(str, system.state)),
            ("Phi_input", "Phi_output", "I_V1_p", "I_R1_p", "I_C1_p"),
        )
        self.assertEqual(len(system.state), len(system.equations))

    def test_dynamic_builtin_models_compile(self) -> None:
        elements = {
            "D1": antispice.Element("1n4148", ("0", "diode")),
            "Q1": antispice.Element("2n3904", ("0", "base", "collector")),
            "M1": antispice.Element("2n7000", ("0", "gate", "drain")),
            "A1": antispice.Element(
                "opamp-slew-limited",
                ("0", "positive-rail", "plus", "minus", "output"),
                {
                    "open_loop_gain": 100_000,
                    "slew_rate": 500_000,
                    "transition_voltage": 0.05,
                    "upper_dropout_voltage": 1.5,
                    "lower_dropout_voltage": 1.5,
                    "upper_input_saturation_voltage": 1.5,
                    "lower_input_saturation_voltage": 1.5,
                },
            ),
        }

        system = compiler.compile_circuit(antispice.Circuit(elements=elements))

        self.assertEqual(len(system.equations), len(system.state))
        equations = " ".join(map(str, system.equations))
        for derivative in ("Phidot_diode", "Phidot_base", "Phidot_gate", "Phidot_output"):
            self.assertIn(derivative, equations)

    def test_port_currents_generate_kcl_at_port_and_reference_nodes(self) -> None:
        model = antispice.Model(
            ports=("reference", "a", "b"),
            parameters=(),
            equations=("U_a - I_a", "U_b - 2 * I_b"),
        )
        circuit = antispice.Circuit(elements={"X1": antispice.Element(model, ("common", "0", "b"))})

        system = compiler.compile_circuit(circuit)
        equations = tuple(map(str, system.equations))

        self.assertEqual(system.layout.potentials, {"common": 0, "b": 1})
        self.assertEqual(equations[0], "-I_X1_a - I_X1_b")
        self.assertEqual(equations[1], "I_X1_b")
        self.assertEqual(equations[2], "-I_X1_a - Phi_common")
        self.assertIn("-2*I_X1_b", equations[3])

    def test_parameters_may_reference_time_and_other_parameters(self) -> None:
        source = antispice.Model(
            ports=("reference", "p"),
            parameters=("frequency", "voltage"),
            equations=("U_p - voltage",),
        )
        circuit = antispice.Circuit(
            elements={
                "V1": antispice.Element(
                    source,
                    ("0", "input"),
                    {
                        "frequency": 1_000,
                        "voltage": "sin(2 * pi * frequency * t)",
                    },
                )
            }
        )

        system = compiler.compile_circuit(circuit)

        self.assertIn("sin", str(system.equations[-1]))
        self.assertIn("t", str(system.equations[-1]))

    def test_reference_node_is_required(self) -> None:
        circuit = antispice.Circuit(elements={"R1": antispice.Element("resistor", ("a", "b"), {"resistance": 10})})

        with self.assertRaisesRegex(ValueError, "reference node '0'"):
            compiler.compile_circuit(circuit)


class RadauTest(unittest.TestCase):
    def test_radau_stages_use_one_third_and_end_of_step_times(self) -> None:
        time, state, derivative = wrenfold.sym.make_symbols(["t", "x", "xdot"])
        system = compiler.EquationSystem(
            time=time,
            state=(state,),
            state_derivative=(derivative,),
            equations=(time,),
            layout=compiler.StateLayout({}, {}),
        )

        step = compiler.discretize_radau_iia(system)

        self.assertTrue(step.equations[0].is_identical_to(step.time_start + step.step_size / 3))
        self.assertTrue(step.equations[1].is_identical_to(step.time_start + step.step_size))


class PublicApiTest(unittest.TestCase):
    def test_public_names_are_available_from_package_root(self) -> None:
        for name in antispice.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(antispice, name))

    def test_low_level_compiler_names_are_not_package_root_exports(self) -> None:
        internal_names = {
            "EquationSystem",
            "WasmGenerator",
            "compile_circuit",
            "generate_python_kernels",
            "radau_memory_layout",
            "transpile_radau_evaluator",
        }
        self.assertTrue(internal_names.isdisjoint(antispice.__all__))


if __name__ == "__main__":
    unittest.main()
