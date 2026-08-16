"""Tests for direct WebAssembly code generation."""

import json
import math
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import wrenfold
from wrenfold import sym

import antispice


def _run_node(module: bytes, program: str) -> object:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("Node.js is required to execute WebAssembly tests")
    with tempfile.TemporaryDirectory() as directory:
        module_path = pathlib.Path(directory) / "module.wasm"
        module_path.write_bytes(module)
        result = subprocess.run(
            [node, "-e", program, str(module_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(result.stdout)


class WasmGeneratorTest(unittest.TestCase):
    """Test the generic Wrenfold WebAssembly generator."""

    def test_generator_is_a_wrenfold_base_generator(self) -> None:
        generator = antispice.WasmGenerator()

        self.assertIsInstance(generator, wrenfold.BaseGenerator)
        with self.assertRaisesRegex(ValueError, "memory_pages"):
            antispice.WasmGenerator(memory_pages=0)

    def test_scalar_function_with_branch_and_math_import(self) -> None:
        def scalar(x: wrenfold.FloatScalar, y: wrenfold.FloatScalar):
            return sym.where(x > 0, sym.sin(x) + 2 * y, sym.sqrt(abs(y)) / 3)

        generator = antispice.WasmGenerator()
        module = wrenfold.generate_function(scalar, generator=generator)

        self.assertIsInstance(module, bytes)
        self.assertEqual(module[:8], b"\x00asm\x01\x00\x00\x00")
        self.assertEqual(generator.abi[0].name, "scalar")
        self.assertEqual(generator.abi[0].return_type, "f64")

        values = _run_node(
            module,
            """
const fs = require("fs");
const imports = {math: {sin: Math.sin}};
WebAssembly.instantiate(fs.readFileSync(process.argv[1]), imports)
  .then(({instance}) => console.log(JSON.stringify([
    instance.exports.scalar(2, 3),
    instance.exports.scalar(-2, 12),
  ])));
""",
        )
        self.assertAlmostEqual(values[0], math.sin(2) + 6)
        self.assertAlmostEqual(values[1], math.sqrt(12) / 3)

    def test_matrix_arguments_use_exported_row_major_memory(self) -> None:
        def scale(x: wrenfold.Vector2):
            return wrenfold.OutputArg(2 * x, "result")

        generator = antispice.WasmGenerator()
        module = wrenfold.generate_function(scale, generator=generator)

        arguments = generator.abi[0].arguments
        self.assertEqual((arguments[0].rows, arguments[0].cols), (2, 1))
        self.assertEqual(arguments[1].direction, "output")

        values = _run_node(
            module,
            """
const fs = require("fs");
WebAssembly.instantiate(fs.readFileSync(process.argv[1]))
  .then(({instance}) => {
    const values = new Float64Array(instance.exports.memory.buffer);
    values[0] = 1.5;
    values[1] = -2;
    instance.exports.scale(0, 16);
    console.log(JSON.stringify([values[2], values[3]]));
  });
""",
        )
        self.assertEqual(values, [3, -4])


class WasmRadauTest(unittest.TestCase):
    """Exercise the WebAssembly backend with the complete Radau solver."""

    def test_generated_solver_integrates_rc_step_response(self) -> None:
        circuit = antispice.Circuit(
            elements={
                "V1": antispice.Element(
                    "voltage_source",
                    ("0", "input"),
                    {"voltage": "where(t > 0, 1, 0)"},
                ),
                "R1": antispice.Element("resistor", ("input", "output"), {"resistance": 1_000.0}),
                "C1": antispice.Element("capacitor", ("0", "output"), {"capacitance": 1e-6}),
            }
        )
        system = antispice.compile_circuit(circuit)
        module = antispice.generate_wasm_radau_solver(system)

        values = _run_node(
            module,
            """
const fs = require("fs");
WebAssembly.instantiate(fs.readFileSync(process.argv[1]))
  .then(({instance}) => {
    const values = new Float64Array(instance.exports.memory.buffer);
    const stage = 0;
    const previous = 80;
    const updated = 120;
    const next = 200;
    for (let step = 0; step < 10; ++step) {
      instance.exports.radau_newton_step(
        stage, step * 1e-4, 1e-4, previous, updated, next);
      values.copyWithin(stage / 8, updated / 8, updated / 8 + 10);
      values.copyWithin(previous / 8, next / 8, next / 8 + 5);
    }
    console.log(JSON.stringify(Array.from(
      values.slice(previous / 8, previous / 8 + 5))));
  });
""",
        )
        output_index = system.layout.potential_index("output")
        self.assertAlmostEqual(values[output_index], 1 - math.exp(-1), delta=1e-5)


if __name__ == "__main__":
    unittest.main()
