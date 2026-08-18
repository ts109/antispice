"""Tests for direct WebAssembly code generation."""

import math
import struct
import unittest

import wrenfold
from wasmtime import Engine, Func, Instance, Memory, Module, Store
from wrenfold import sym

import antispice
from antispice import compiler, wasm, wasm_target


def _instantiate(module: bytes, imports: dict[str, object] | None = None) -> tuple[Store, Instance]:
    engine = Engine()
    compiled = Module(engine, module)
    store = Store(engine)
    functions = []
    for imported in compiled.imports:
        callback = (imports or {})[f"{imported.module}.{imported.name}"]
        functions.append(Func(store, imported.type, callback))
    return store, Instance(store, compiled, functions)


def _write_f64(memory: Memory, store: Store, offset: int, values: list[float]) -> None:
    memory.write(store, struct.pack(f"<{len(values)}d", *values), offset)


def _read_f64(memory: Memory, store: Store, offset: int, count: int) -> list[float]:
    return list(struct.unpack(f"<{count}d", memory.read(store, offset, offset + count * 8)))


class WasmGeneratorTest(unittest.TestCase):
    """Test the generic Wrenfold WebAssembly generator."""

    def test_generator_is_a_wrenfold_base_generator(self) -> None:
        generator = wasm.WasmGenerator()

        self.assertIsInstance(generator, wrenfold.BaseGenerator)
        with self.assertRaisesRegex(ValueError, "memory_pages"):
            wasm.WasmGenerator(memory_pages=0)

    def test_scalar_function_with_branch_and_math_import(self) -> None:
        def scalar(x: wrenfold.FloatScalar, y: wrenfold.FloatScalar) -> wrenfold.FloatScalar:
            return sym.where(x > 0, sym.sin(x) + 2 * y, sym.sqrt(abs(y)) / 3)

        generator = wasm.WasmGenerator()
        module = wrenfold.generate_function(scalar, generator=generator)

        self.assertIsInstance(module, bytes)
        self.assertEqual(module[:8], b"\x00asm\x01\x00\x00\x00")
        self.assertEqual(generator.abi[0].name, "scalar")
        self.assertEqual(generator.abi[0].return_type, "f64")

        store, instance = _instantiate(module, {"math.sin": math.sin})
        scalar = instance.exports(store)["scalar"]
        values = [scalar(store, 2.0, 3.0), scalar(store, -2.0, 12.0)]
        self.assertAlmostEqual(values[0], math.sin(2) + 6)
        self.assertAlmostEqual(values[1], math.sqrt(12) / 3)

    def test_matrix_arguments_use_exported_row_major_memory(self) -> None:
        def scale(x: wrenfold.Vector2) -> wrenfold.OutputArg:
            return wrenfold.OutputArg(2 * x, "result")

        generator = wasm.WasmGenerator()
        module = wrenfold.generate_function(scale, generator=generator)

        arguments = generator.abi[0].arguments
        self.assertEqual((arguments[0].rows, arguments[0].cols), (2, 1))
        self.assertEqual(arguments[1].direction, "output")

        store, instance = _instantiate(module)
        exports = instance.exports(store)
        memory = exports["memory"]
        _write_f64(memory, store, 0, [1.5, -2])
        exports["scale"](store, 0, 16)
        values = _read_f64(memory, store, 16, 2)
        self.assertEqual(values, [3, -4])


class WasmRadauTest(unittest.TestCase):
    """Exercise the WebAssembly backend with the complete Radau solver."""

    def test_dense_lu_uses_partial_pivoting_and_reports_singular_matrices(self) -> None:
        """The native numerical solver swaps rows and returns useful status codes."""
        module = wasm.WasmGenerator().generate(wasm.dense_lu_solve_function())
        store, instance = _instantiate(module)
        exports = instance.exports(store)
        memory = exports["memory"]
        solve = exports["dense_lu_solve"]
        _write_f64(memory, store, 0, [0, 2, 1, 3, 4, 7])
        self.assertEqual(solve(store, 0, 32, 2, 1e-14), 0)
        solution = _read_f64(memory, store, 32, 2)
        self.assertAlmostEqual(solution[0], 1)
        self.assertAlmostEqual(solution[1], 2)

        _write_f64(memory, store, 0, [1, 2, 2, 4, 1, 2])
        self.assertEqual(solve(store, 0, 32, 2, 1e-14), 1)

        _write_f64(memory, store, 0, [math.nan, 0, 0, 1, 1, 1])
        self.assertEqual(solve(store, 0, 32, 2, 1e-14), 2)

    def test_stationary_evaluator_finds_dc_operating_point(self) -> None:
        """Newton iterations solve the original DAE with all derivatives zero."""
        circuit = antispice.Circuit(
            elements={
                "V1": antispice.Element("voltage-source", ("0", "output"), {"voltage": 5}),
                "R1": antispice.Element("resistor", ("output", "0"), {"resistance": 1_000}),
            }
        )
        system = compiler.compile_circuit(circuit)
        layout = compiler.radau_memory_layout(system)
        store, instance = _instantiate(wasm_target.generate_wasm_solver(system))
        exports = instance.exports(store)
        memory = exports["memory"]
        state = [0.0] * len(system.state)

        for _ in range(10):
            _write_f64(memory, store, layout.previous_state, state)
            exports["stationary_evaluate"](
                store,
                layout.previous_state,
                0.0,
                layout.residual,
                layout.jacobian,
            )
            self.assertEqual(exports["dense_lu_solve"](store, layout.jacobian, layout.residual, len(state), 1e-14), 0)
            correction = _read_f64(memory, store, layout.residual, len(state))
            updated = [value - delta for value, delta in zip(state, correction, strict=True)]
            if max(abs(after - before) for after, before in zip(updated, state, strict=True)) <= 1e-10:
                state = updated
                break
            state = updated

        self.assertAlmostEqual(state[system.layout.potential_index("output")], 5)

    def test_auxiliary_evaluator_observes_state_without_solver_overhead(self) -> None:
        """Named model expressions are evaluated through the generated WASM ABI."""
        model = antispice.Model(
            ports=("ref", "p"),
            parameters=("gain",),
            equations=("I_p",),
            auxiliaries={"scaled": "gain * U_p"},
        )
        system = compiler.compile_circuit(antispice.Circuit(elements={"X1": antispice.Element(model, ("0", "out"), {"gain": 3})}))
        layout = compiler.radau_memory_layout(system)
        store, instance = _instantiate(wasm_target.generate_wasm_solver(system))
        exports = instance.exports(store)
        memory = exports["memory"]
        state = [0.0] * len(system.state)
        state[system.layout.potential_index("out")] = 2.5
        _write_f64(memory, store, layout.previous_state, state)
        _write_f64(memory, store, layout.stage_derivatives, [0.0] * len(system.state))

        exports["evaluate_auxiliaries"](
            store,
            layout.previous_state,
            layout.stage_derivatives,
            0.0,
            layout.auxiliary_values,
        )

        self.assertEqual(_read_f64(memory, store, layout.auxiliary_values, 1), [7.5])

    def test_generated_solver_integrates_rc_step_response(self) -> None:
        circuit = antispice.Circuit(
            elements={
                "V1": antispice.Element(
                    "voltage-source",
                    ("0", "input"),
                    {"voltage": "where(t > 0, 1, 0)"},
                ),
                "R1": antispice.Element("resistor", ("input", "output"), {"resistance": 1_000.0}),
                "C1": antispice.Element("capacitor", ("0", "output"), {"capacitance": 1e-6}),
            }
        )
        system = compiler.compile_circuit(circuit)
        module = wasm_target.generate_wasm_solver(system)

        store, instance = _instantiate(module)
        exports = instance.exports(store)
        memory = exports["memory"]
        evaluate = exports["radau_evaluate"]
        solve = exports["dense_lu_solve"]
        layout = compiler.radau_memory_layout(system)
        state_size = len(system.state)
        newton_size = 2 * state_size
        stage = [0.0] * newton_size
        state = [0.0] * state_size
        step_size = 1e-4

        for step_index in range(10):
            _write_f64(memory, store, layout.previous_state, state)
            for _ in range(10):
                _write_f64(memory, store, layout.stage_derivatives, stage)
                evaluate(
                    store,
                    layout.stage_derivatives,
                    step_index * step_size,
                    step_size,
                    layout.previous_state,
                    layout.residual,
                    layout.jacobian,
                )
                self.assertEqual(solve(store, layout.jacobian, layout.residual, newton_size, 1e-14), 0)
                correction = _read_f64(memory, store, layout.residual, newton_size)
                updated = [value - delta for value, delta in zip(stage, correction, strict=True)]
                error = max(abs(after - before) for after, before in zip(updated, stage, strict=True))
                stage = updated
                if error <= 1e-10:
                    break
            state = [state[index] + step_size * (3 / 4 * stage[index] + 1 / 4 * stage[state_size + index]) for index in range(state_size)]

        output_index = system.layout.potential_index("output")
        self.assertAlmostEqual(state[output_index], 1 - math.exp(-1), delta=1e-5)

    def test_high_level_target_returns_a_complete_host_artifact(self) -> None:
        circuit = antispice.Circuit(elements={"R1": antispice.Element("resistor", ("0", "out"), {"resistance": 1_000.0})})

        artifact = antispice.compile_wasm(
            circuit,
            ac_cases=[{"reference": "AC1", "type": "current", "net": "out"}],
        )

        self.assertTrue(artifact.module.startswith(b"\0asm"))
        self.assertIn("export class AntispiceSolver", artifact.javascript)
        self.assertIn('"reference": "AC1"', artifact.javascript)
        self.assertEqual(artifact.layout.potentials, {"out": 0})
        self.assertEqual(artifact.layout.currents, {"R1": {"p": 1}})


if __name__ == "__main__":
    unittest.main()
