"""Tests for the optional generated native Python solver backend."""

import math
import unittest
from unittest.mock import patch

try:
    import numpy
except ImportError:  # pragma: no cover - exercised by optional-dependency users
    numpy = None

import antispice


def make_rc_circuit() -> antispice.Circuit:
    """Construct a unit-step RC low-pass circuit."""
    return antispice.Circuit(
        elements={
            "V1": antispice.Element("voltage-source", ("0", "input"), {"voltage": "where(t > 0, 1, 0)"}),
            "R1": antispice.Element("resistor", ("input", "output"), {"resistance": 1_000.0}),
            "C1": antispice.Element("capacitor", ("0", "output"), {"capacitance": 1e-6}),
        }
    )


@unittest.skipIf(numpy is None, "NumPy is an optional dependency")
class PythonSolverTest(unittest.TestCase):
    """Exercise the generated kernels through the stateless high-level API."""

    def test_compilation_exposes_native_generated_evaluators(self) -> None:
        solver = antispice.compile_python(make_rc_circuit())

        self.assertIn("def radau_evaluate", solver.generated_source)
        self.assertIn("def stationary_evaluate", solver.generated_source)
        self.assertIn("def ac_linearize", solver.generated_source)
        self.assertEqual(solver.layout.state_size, 5)

    def test_operating_point_and_adaptive_transient_match_rc_response(self) -> None:
        solver = antispice.compile_python(make_rc_circuit())

        operating_point = solver.operating_point(0.0)
        result = solver.transient(
            start_time=0.0,
            end_time=1e-3,
            minimum_step_size=1e-7,
            maximum_step_size=1e-4,
            relative_tolerance=1e-5,
        )

        self.assertEqual(operating_point.potential("output"), 0.0)
        self.assertEqual(result.times[0], 0.0)
        self.assertEqual(result.times[-1], 1e-3)
        self.assertEqual(result.sample_count, result.accepted_steps + 1)
        self.assertAlmostEqual(result.potential("output")[-1], 1 - math.exp(-1), delta=1e-5)
        self.assertTrue(numpy.shares_memory(result.potential("output"), result.states))

    def test_trust_radius_converges_difficult_bjt_operating_point(self) -> None:
        circuit = antispice.Circuit(
            elements={
                "I1": antispice.Element("current-source", ("0", "bias"), {"current": 1e-3}),
                "Q1": antispice.Element("2n3904", ("0", "bias", "bias")),
            }
        )

        result = antispice.compile_python(circuit).operating_point()

        self.assertLess(result.residual_norm, 1e-10)
        self.assertGreater(result.trust_limited_steps, 0)
        self.assertGreater(result.potential("bias"), 0.5)

    def test_adaptive_transient_ignores_algebraic_current_discontinuities(self) -> None:
        """A stepped source must not make algebraic branch currents control the timestep."""
        circuit = antispice.Circuit(
            elements={
                "L1": antispice.Element(
                    "tapped-inductor",
                    ("0", "tap", "tank"),
                    {"first_inductance": 1e-6, "second_inductance": 1e-6, "coupling_factor": 1},
                ),
                "C1": antispice.Element("capacitor", ("0", "tank"), {"capacitance": 1e-6}),
                "Q1": antispice.Element("2n7000", ("source", "gate", "supply")),
                "C2": antispice.Element("capacitor", ("tank", "gate"), {"capacitance": 1e-9}),
                "R1": antispice.Element("resistor", ("0", "gate"), {"resistance": 100e3}),
                "R2": antispice.Element("resistor", ("gate", "supply"), {"resistance": 100e3}),
                "R3": antispice.Element("resistor", ("tap", "source"), {"resistance": 1e3}),
                "V1": antispice.Element("voltage-source", ("0", "supply"), {"voltage": "where(t > 0, 10, 0)"}),
            }
        )
        solver = antispice.compile_python(circuit)

        result = solver.transient(
            start_time=0,
            end_time=10e-6,
            minimum_step_size=1e-10,
            maximum_step_size=1e-5,
            relative_tolerance=1e-4,
            voltage_absolute_tolerance=1e-7,
            current_absolute_tolerance=1e-10,
        )

        self.assertEqual(result.times[-1], 10e-6)
        self.assertAlmostEqual(result.potential("supply")[-1], 10)
        self.assertLess(len(solver.layout.differential_states), solver.layout.state_size)

    def test_opamp_unity_buffer_tracks_a_ten_millivolt_step(self) -> None:
        amplitude = 1e-2
        circuit = antispice.Circuit(
            elements={
                "VNEG": antispice.Element("voltage-source", ("0", "negative-rail"), {"voltage": -15}),
                "VPOS": antispice.Element("voltage-source", ("0", "positive-rail"), {"voltage": 15}),
                "VIN": antispice.Element("voltage-source", ("0", "input"), {"voltage": f"where(t > 0, {amplitude}, 0)"}),
                "A1": antispice.Element(
                    "opamp",
                    ("negative-rail", "positive-rail", "input", "output", "output"),
                    {
                        "open_loop_gain": 100_000,
                        "slew_rate": 1e6,
                        "transition_voltage": 0.1,
                        "upper_dropout_voltage": 0,
                        "lower_dropout_voltage": 0,
                        "upper_input_saturation_voltage": 0,
                        "lower_input_saturation_voltage": 0,
                    },
                ),
            }
        )
        solver = antispice.compile_python(circuit)

        operating_point = solver.operating_point(0)
        result = solver.transient(
            start_time=0,
            end_time=2e-6,
            minimum_step_size=1e-10,
            maximum_step_size=1e-7,
            relative_tolerance=1e-5,
        )

        self.assertAlmostEqual(operating_point.potential("output"), 0, delta=1e-10)
        self.assertAlmostEqual(result.potential("output")[-1], amplitude, delta=amplitude * 1e-4)

    def test_transient_storage_grows_geometrically_beyond_initial_capacity(self) -> None:
        circuit = antispice.Circuit(elements={"R1": antispice.Element("resistor", ("0", "output"), {"resistance": 1_000.0})})
        solver = antispice.compile_python(circuit)

        result = solver.transient(
            start_time=0.0,
            end_time=0.03,
            minimum_step_size=1e-4,
            maximum_step_size=1e-4,
        )

        self.assertEqual(result.accepted_steps, 300)
        self.assertEqual(result.sample_count, 301)
        self.assertEqual(result.states.shape, (301, solver.layout.state_size))
        self.assertGreaterEqual(result.states.base.shape[0], 301)

    def test_results_provide_port_current_voltage_and_auxiliary_access(self) -> None:
        circuit = antispice.Circuit(elements={"Q1": antispice.Element("2n3904", ("0", "base", "collector"))})
        solver = antispice.compile_python(circuit)
        operating_point = solver.operating_point(0.0)

        self.assertEqual(operating_point.port_voltage("Q1", "B"), operating_point.potential("base"))
        self.assertEqual(
            operating_point.current("Q1", "E"),
            -operating_point.current("Q1", "B") - operating_point.current("Q1", "C"),
        )
        self.assertEqual(operating_point.auxiliary("Q1", "v_be"), 0.0)

    def test_ac_current_input_matches_parallel_rc_impedance(self) -> None:
        solver = antispice.compile_python(make_rc_circuit())
        operating_point = solver.operating_point(0.0)
        frequency = 1_000.0

        result = solver.linearize(operating_point.state).sweep(
            antispice.ACCurrentInput("output"),
            numpy.array([frequency]),
        )

        expected = 1 / (1 / 1_000 + 2j * math.pi * frequency * 1e-6)
        self.assertAlmostEqual(result.potential("output")[0], expected)

    def test_ac_convenience_method_uses_the_same_compiled_system(self) -> None:
        solver = antispice.compile_python(make_rc_circuit())

        result = solver.ac(
            antispice.ACCurrentInput("output"),
            numpy.array([1_000.0]),
            time=0.0,
        )

        self.assertEqual(result.sample_count, 1)

    def test_reusing_workspace_does_not_alias_returned_results(self) -> None:
        solver = antispice.compile_python(make_rc_circuit())
        workspace = solver.create_workspace()

        first = solver.operating_point(0.0, workspace=workspace)
        second = solver.operating_point(1.0, workspace=workspace)

        self.assertEqual(first.potential("input"), 0.0)
        self.assertEqual(second.potential("input"), 1.0)
        self.assertEqual(first.potential("input"), 0.0)

    def test_missing_numpy_has_an_actionable_optional_dependency_error(self) -> None:
        with (
            patch("antispice.python_solver.importlib.import_module", side_effect=ImportError),
            self.assertRaisesRegex(antispice.PythonBackendUnavailable, r"antispice\[python\]"),
        ):
            antispice.compile_python(make_rc_circuit())


if __name__ == "__main__":
    unittest.main()
