"""Tests for the generated high-level JavaScript solver wrapper."""

import json
import math
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import antispice
from antispice import compiler, javascript, wasm_target


def _rc_system() -> compiler.EquationSystem:
    circuit = antispice.Circuit(
        elements={
            "V1": antispice.Element(
                "voltage-source",
                ("0", "input"),
                {"voltage": "where(t > 0, 1, 0)"},
            ),
            "R1": antispice.Element(
                "resistor",
                ("input", "output"),
                {"resistance": 1_000.0},
            ),
            "C1": antispice.Element(
                "capacitor",
                ("0", "output"),
                {"capacitance": 1e-6},
            ),
        }
    )
    return compiler.compile_circuit(circuit)


def _bjt_step_system() -> compiler.EquationSystem:
    """Construct the nonlinear step-response circuit from the web editor."""
    circuit = antispice.Circuit(
        elements={
            "V1": antispice.Element("voltage-source", ("0", "supply"), {"voltage": "where(t > 0, 10, 0)"}),
            "R1": antispice.Element("resistor", ("supply", "collector"), {"resistance": 1e3}),
            "R2": antispice.Element("resistor", ("0", "emitter"), {"resistance": 1e3}),
            "R3": antispice.Element("resistor", ("base", "supply"), {"resistance": 200e3}),
            "R4": antispice.Element("resistor", ("0", "base"), {"resistance": 50e3}),
            "C1": antispice.Element("capacitor", ("base", "collector"), {"capacitance": 100e-6}),
            "Q1": antispice.Element("2n3904", ("emitter", "base", "collector"), {}),
        }
    )
    return compiler.compile_circuit(circuit)


def _bjt_sine_system() -> compiler.EquationSystem:
    """Construct the collector-feedback amplifier used for predictor fallback."""
    circuit = antispice.Circuit(
        elements={
            "V1": antispice.Element("voltage-source", ("0", "supply"), {"voltage": 10}),
            "V2": antispice.Element("voltage-source", ("0", "input"), {"voltage": "sin(1000 * t)"}),
            "R1": antispice.Element("resistor", ("collector", "supply"), {"resistance": 1e3}),
            "R2": antispice.Element("resistor", ("base", "collector"), {"resistance": 200e3}),
            "C1": antispice.Element("capacitor", ("input", "base"), {"capacitance": 10e-6}),
            "Q1": antispice.Element("2n3904", ("0", "base", "collector"), {}),
        }
    )
    return compiler.compile_circuit(circuit)


class JavaScriptWrapperTest(unittest.TestCase):
    """Exercise both views and the integration interface in Node.js."""

    def test_invalid_export_names_are_rejected(self) -> None:
        system = _rc_system()

        with self.assertRaisesRegex(ValueError, "class name"):
            javascript.generate_javascript_radau_wrapper(system, class_name="not valid")
        with self.assertRaisesRegex(ValueError, "function name"):
            javascript.generate_javascript_radau_wrapper(system, function_name="not-valid")

    def test_adaptive_error_uses_only_differential_states(self) -> None:
        wrapper = javascript.generate_javascript_radau_wrapper(_rc_system())

        self.assertIn("const DIFFERENTIAL_STATE_INDICES = Object.freeze([1]);", wrapper)
        self.assertEqual(wrapper.count("for (const index of DIFFERENTIAL_STATE_INDICES)"), 1)
        self.assertIn("differentialStateIndices: DIFFERENTIAL_STATE_INDICES", wrapper)

    def test_operating_point_uses_residual_backtracking(self) -> None:
        """Generated initialization rejects steps until the residual decreases."""
        wrapper = javascript.generate_javascript_radau_wrapper(_rc_system())

        self.assertIn("candidateNorm < norm", wrapper)
        self.assertIn("multiplier *= 0.5", wrapper)
        self.assertIn("minimumStepMultiplier = 2 ** -20", wrapper)
        self.assertIn("Operating-point backtracking failed", wrapper)

    def test_transient_newton_skips_solved_residuals(self) -> None:
        """A stationary circuit does not factor an unnecessary Radau Jacobian."""
        wrapper = javascript.generate_javascript_radau_wrapper(_rc_system())

        convergence = wrapper.index("residualNorm <= residualTolerance")
        factorization = wrapper.index("const status = this._linearSolve", convergence)
        self.assertLess(convergence, factorization)

    def test_transient_newton_uses_residual_backtracking(self) -> None:
        """Every implicit step rejects Newton updates that worsen its residual."""
        wrapper = javascript.generate_javascript_radau_wrapper(_rc_system())

        self.assertIn("candidateNorm < residualNorm", wrapper)
        self.assertIn("this._transientPreviousStage", wrapper)
        self.assertIn("this._transientCorrection", wrapper)
        self.assertIn("Transient Newton backtracking failed", wrapper)
        self.assertIn("if (residualNorm <= residualTolerance)", wrapper)
        self.assertIn("if (attempt === 1) this.vectors.stageDerivatives.fill(0)", wrapper)

    def test_convergence_errors_include_simulation_time(self) -> None:
        """Generated operating-point and integration errors identify their time."""
        wrapper = javascript.generate_javascript_radau_wrapper(_rc_system())

        self.assertIn("Operating-point Newton iteration did not converge at t=${time}", wrapper)
        self.assertIn("Transient solve failed at t=${time + stepSize}", wrapper)
        self.assertIn("Newton iteration did not converge at t=${time + stepSize}", wrapper)

    def test_tolerance_options_are_validated_and_forwarded(self) -> None:
        """IC, transient Newton, and adaptive control share explicit options."""
        wrapper = javascript.generate_javascript_radau_wrapper(_rc_system())

        self.assertIn("residualTolerance must be positive and finite", wrapper)
        self.assertIn("relativeTolerance must be positive and finite", wrapper)
        self.assertIn("voltageAbsoluteTolerance must be positive and finite", wrapper)
        self.assertIn("currentAbsoluteTolerance must be positive and finite", wrapper)
        self.assertIn("...stepOptions", wrapper)

    def test_wrapper_can_collect_columnar_results(self) -> None:
        """Plotting output uses contiguous typed arrays instead of snapshot objects."""
        wrapper = javascript.generate_javascript_radau_wrapper(_rc_system())

        self.assertIn("integrateArrays(options)", wrapper)
        self.assertIn("const times = new Float64Array(capacity)", wrapper)
        self.assertIn("const states = new Float64Array(capacity * STATE_SIZE)", wrapper)
        self.assertNotIn("for (const sample of this.integrate(options))", wrapper)
        self.assertIn("states.set(this.vectors.state", wrapper)
        self.assertIn("times.subarray(0, sampleCount)", wrapper)
        self.assertIn("const targetTime = Math.min", wrapper)

    def test_ac_wrapper_exposes_independent_frequency_sweeps(self) -> None:
        """Dedicated marker metadata selects one separately solved AC right-hand side."""
        cases = [
            {"reference": "AC1", "type": "voltage", "net": "input"},
            {"reference": "AC2", "type": "current", "net": "output"},
        ]

        wrapper = javascript.generate_javascript_radau_wrapper(_rc_system(), ac_cases=cases)

        self.assertIn('"reference": "AC1", "type": "voltage", "net": "input"', wrapper)
        self.assertIn('"reference": "AC2", "type": "current", "net": "output"', wrapper)
        self.assertIn("initializeAC(time, options = {})", wrapper)
        self.assertIn("solveAC(caseIndex, frequency", wrapper)
        self.assertIn("sweepAC(caseIndex, frequencies", wrapper)
        self.assertIn("const omega = 2 * Math.PI * frequency", wrapper)

    def test_ac_current_marker_solves_rc_impedance(self) -> None:
        """A marker RHS excites the linearized circuit while its source remains bias-only."""
        node = shutil.which("node")
        if node is None:
            message = "Node.js is required to execute JavaScript tests"
            raise unittest.SkipTest(message)
        system = _rc_system()
        cases = [{"reference": "AC1", "type": "current", "net": "output"}]
        module = wasm_target.generate_wasm_solver(system)
        wrapper = javascript.generate_javascript_radau_wrapper(system, ac_cases=cases)
        output_index = system.layout.potential_index("output")
        runner = f"""
import fs from "node:fs";
import {{AntispiceSolver}} from "./wrapper.mjs";
const solver = await AntispiceSolver.instantiate(fs.readFileSync("solver.wasm"));
solver.initializeAC(0);
const result = solver.solveAC(0, 1000);
const omega = 2 * Math.PI * 1000;
const expectedReal = 0.001 / (0.001 ** 2 + (omega * 1e-6) ** 2);
const expectedImaginary = -omega * 1e-6 / (0.001 ** 2 + (omega * 1e-6) ** 2);
if (Math.abs(result.real[{output_index}] - expectedReal) > 1e-9) process.exit(1);
if (Math.abs(result.imaginary[{output_index}] - expectedImaginary) > 1e-9) process.exit(2);
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "solver.wasm").write_bytes(module)
            (path / "wrapper.mjs").write_text(wrapper)
            (path / "runner.mjs").write_text(runner)
            subprocess.run([node, "runner.mjs"], cwd=path, check=True)

    def test_adaptive_wrapper_uses_step_doubling_without_resampling(self) -> None:
        """Adaptive output contains accepted endpoints at their actual times."""
        node = shutil.which("node")
        if node is None:
            message = "Node.js is required to execute JavaScript tests"
            raise unittest.SkipTest(message)

        system = _rc_system()
        module = wasm_target.generate_wasm_solver(system)
        wrapper = javascript.generate_javascript_radau_wrapper(system)
        self.assertIn("Math.sqrt(minimumStepSize * maximumStepSize)", wrapper)
        runner = """
import fs from "node:fs";
import {AntispiceSolver} from "./wrapper.mjs";
const solver = await AntispiceSolver.instantiate(fs.readFileSync("solver.wasm"));
solver.initializeOperatingPoint(0);
const result = solver.integrateAdaptiveArrays({
  startTime: 0,
  endTime: 0.01,
  minimumStepSize: 1e-7,
  maximumStepSize: 1e-3,
  relativeTolerance: 1e-5,
});
if (result.times[0] !== 0 || result.times.at(-1) !== 0.01) process.exit(1);
if (!(result.acceptedSteps > 0) || result.sampleCount !== result.acceptedSteps + 1) process.exit(2);
let unequal = false;
for (let i = 2; i < result.sampleCount; ++i) {
  const previous = result.times[i - 1] - result.times[i - 2];
  const current = result.times[i] - result.times[i - 1];
  if (Math.abs(previous - current) > 1e-15) unequal = true;
}
if (!unequal) process.exit(3);
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "solver.wasm").write_bytes(module)
            (path / "wrapper.mjs").write_text(wrapper)
            (path / "runner.mjs").write_text(runner)
            subprocess.run([node, "runner.mjs"], cwd=path, check=True)

    def test_structured_and_flat_views_alias_and_integrate(self) -> None:
        node = shutil.which("node")
        if node is None:
            message = "Node.js is required to execute JavaScript tests"
            raise unittest.SkipTest(message)

        system = _rc_system()
        module = wasm_target.generate_wasm_solver(system)
        wrapper = javascript.generate_javascript_radau_wrapper(system)
        input_index = system.layout.potential_index("input")

        runner = f"""
import fs from "node:fs";
import {{AntispiceSolver, circuitLayout}} from "./wrapper.mjs";

const solver = await AntispiceSolver.instantiate(fs.readFileSync(process.argv[2]));
solver.state.setPotential("input", 0.25);
const namedToFlat = solver.vectors.state[{input_index}];
solver.vectors.state[{input_index}] = 0.5;
const flatToNamed = solver.state.potential("input");
solver.state.setCurrent("R1", "p", 0.125);
const namedCurrent = solver.state.current("R1", "p");
solver.reset();

const samples = Array.from(solver.integrate({{
  startTime: 0,
  endTime: 1e-3,
  stepSize: 1e-4,
}}));
console.log(JSON.stringify({{
  namedToFlat,
  flatToNamed,
  namedCurrent,
  ground: solver.state.potential("0"),
  stateSize: circuitLayout.stateSize,
  sampleCount: samples.length,
  firstOutput: samples[0].state.potentials.output,
  lastOutput: samples.at(-1).state.potentials.output,
  lastTime: samples.at(-1).time,
}}));
"""

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "solver.wasm").write_bytes(module)
            (path / "wrapper.mjs").write_text(wrapper)
            (path / "runner.mjs").write_text(runner)
            result = subprocess.run(
                [node, "runner.mjs", "solver.wasm"],
                cwd=path,
                check=True,
                capture_output=True,
                text=True,
            )

        values = json.loads(result.stdout)
        self.assertEqual(values["namedToFlat"], 0.25)
        self.assertEqual(values["flatToNamed"], 0.5)
        self.assertEqual(values["namedCurrent"], 0.125)
        self.assertEqual(values["ground"], 0)
        self.assertEqual(values["stateSize"], len(system.state))
        self.assertEqual(values["sampleCount"], 11)
        self.assertEqual(values["firstOutput"], 0)
        self.assertAlmostEqual(values["lastTime"], 1e-3)
        self.assertAlmostEqual(values["lastOutput"], 1 - math.exp(-1), delta=1e-5)

    def test_bjt_step_response_uses_end_stage_as_next_predictor(self) -> None:
        """A discontinuous first step must not poison the following stage guess."""
        node = shutil.which("node")
        if node is None:
            message = "Node.js is required to execute JavaScript tests"
            raise unittest.SkipTest(message)

        system = _bjt_step_system()
        module = wasm_target.generate_wasm_solver(system)
        wrapper = javascript.generate_javascript_radau_wrapper(system)
        runner = """
import fs from "node:fs";
import {AntispiceSolver} from "./wrapper.mjs";
const solver = await AntispiceSolver.instantiate(fs.readFileSync("solver.wasm"));
solver.initializeOperatingPoint(0);
const first = solver.step(0, 1e-5, {maxIterations: 20});
const second = solver.step(1e-5, 1e-5, {maxIterations: 20});
if (!first.converged || !second.converged) process.exit(1);
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "solver.wasm").write_bytes(module)
            (path / "wrapper.mjs").write_text(wrapper)
            (path / "runner.mjs").write_text(runner)
            subprocess.run([node, "runner.mjs"], cwd=path, check=True)

    def test_bjt_sine_response_retries_without_a_bad_predictor(self) -> None:
        """A rapidly changing endpoint predictor falls back to a zero start."""
        node = shutil.which("node")
        if node is None:
            message = "Node.js is required to execute JavaScript tests"
            raise unittest.SkipTest(message)

        system = _bjt_sine_system()
        module = wasm_target.generate_wasm_solver(system)
        wrapper = javascript.generate_javascript_radau_wrapper(system)
        runner = """
import fs from "node:fs";
import {AntispiceSolver} from "./wrapper.mjs";
const solver = await AntispiceSolver.instantiate(fs.readFileSync("solver.wasm"));
solver.initializeOperatingPoint(0);
const result = solver.integrateArrays({startTime: 0, endTime: 0.01, stepSize: 1e-4, maxIterations: 20});
if (result.sampleCount !== 101) process.exit(1);
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "solver.wasm").write_bytes(module)
            (path / "wrapper.mjs").write_text(wrapper)
            (path / "runner.mjs").write_text(runner)
            subprocess.run([node, "runner.mjs"], cwd=path, check=True)


if __name__ == "__main__":
    unittest.main()
