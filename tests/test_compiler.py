"""Tests for circuit interpretation and Radau-IIA solver generation."""

import math
import unittest

import numpy
import wrenfold

import antispice


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
        system = antispice.compile_circuit(antispice.Circuit(elements={"X1": antispice.Element(model, ("0", "n"), {"gain": 3})}))

        equation = str(system.equations[-1])
        self.assertNotIn("output", equation)
        self.assertNotIn("squared", equation)
        self.assertIn("Phi_n**2", equation)
        self.assertEqual(len(system.state), 2)

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
                    antispice.compile_circuit(circuit)

    def test_builtin_transistor_equations_compile_for_both_polarities(self) -> None:
        for polarity in (-1, 1):
            with self.subTest(device="bjt", polarity=polarity):
                system = antispice.compile_circuit(
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
                system = antispice.compile_circuit(
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
                                },
                            )
                        }
                    )
                )
                self.assertEqual(len(system.equations), len(system.state))

    def test_compiler_creates_deterministic_state_layout(self) -> None:
        system = antispice.compile_circuit(make_rc_step_circuit())

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

    def test_port_currents_generate_kcl_at_port_and_reference_nodes(self) -> None:
        model = antispice.Model(
            ports=("reference", "a", "b"),
            parameters=(),
            equations=("U_a - I_a", "U_b - 2 * I_b"),
        )
        circuit = antispice.Circuit(elements={"X1": antispice.Element(model, ("common", "0", "b"))})

        system = antispice.compile_circuit(circuit)
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

        system = antispice.compile_circuit(circuit)

        self.assertIn("sin", str(system.equations[-1]))
        self.assertIn("t", str(system.equations[-1]))

    def test_reference_node_is_required(self) -> None:
        circuit = antispice.Circuit(elements={"R1": antispice.Element("resistor", ("a", "b"), {"resistance": 10})})

        with self.assertRaisesRegex(ValueError, "reference node '0'"):
            antispice.compile_circuit(circuit)


class RadauTest(unittest.TestCase):
    def test_radau_stages_use_one_third_and_end_of_step_times(self) -> None:
        time, state, derivative = wrenfold.sym.make_symbols(["t", "x", "xdot"])
        system = antispice.EquationSystem(
            time=time,
            state=(state,),
            state_derivative=(derivative,),
            equations=(time,),
            layout=antispice.StateLayout({}, {}),
        )

        step = antispice.discretize_radau_iia(system)

        self.assertTrue(step.equations[0].is_identical_to(step.time_start + step.step_size / 3))
        self.assertTrue(step.equations[1].is_identical_to(step.time_start + step.step_size))

    def test_generated_python_solver_integrates_rc_step_response(self) -> None:
        system = antispice.compile_circuit(make_rc_step_circuit())
        source = antispice.generate_python_radau_solver(system)
        namespace: dict[str, object] = {}
        exec(source, namespace)
        solver = namespace["radau_newton_step"]

        state = numpy.zeros(len(system.state))
        stage_derivatives = numpy.zeros(2 * len(system.state))
        step_size = 1e-4
        for step_index in range(10):
            stage_derivatives, state = solver(
                stage_derivatives,
                step_index * step_size,
                step_size,
                state,
            )

        output = state.reshape(-1)[system.layout.potential_index("output")]
        expected = 1 - math.exp(-1)
        self.assertAlmostEqual(output, expected, delta=1e-5)


class PublicApiTest(unittest.TestCase):
    def test_public_names_are_available_from_package_root(self) -> None:
        for name in antispice.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(antispice, name))


if __name__ == "__main__":
    unittest.main()
