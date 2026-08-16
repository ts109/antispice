"""Tests for the generated high-level JavaScript solver wrapper."""

import json
import math
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import antispice


def _rc_system() -> antispice.EquationSystem:
    circuit = antispice.Circuit(
        elements={
            "V1": antispice.Element(
                "voltage_source",
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
    return antispice.compile_circuit(circuit)


class JavaScriptWrapperTest(unittest.TestCase):
    """Exercise both views and the integration interface in Node.js."""

    def test_invalid_export_names_are_rejected(self) -> None:
        system = _rc_system()

        with self.assertRaisesRegex(ValueError, "class name"):
            antispice.generate_javascript_radau_wrapper(system, class_name="not valid")
        with self.assertRaisesRegex(ValueError, "function name"):
            antispice.generate_javascript_radau_wrapper(system, function_name="not-valid")

    def test_structured_and_flat_views_alias_and_integrate(self) -> None:
        node = shutil.which("node")
        if node is None:
            raise unittest.SkipTest("Node.js is required to execute JavaScript tests")

        system = _rc_system()
        module = antispice.generate_wasm_radau_solver(system)
        wrapper = antispice.generate_javascript_radau_wrapper(system)
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


if __name__ == "__main__":
    unittest.main()
