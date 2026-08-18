"""High-level WebAssembly compilation target for Antispice circuits."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from .circuit import Circuit
from .compiler import (
    EquationSystem,
    compile_circuit,
    discretize_radau_iia,
    radau_memory_layout,
    transpile_ac_auxiliary_linearizer,
    transpile_ac_linearizer,
    transpile_auxiliary_evaluator,
    transpile_radau_evaluator,
    transpile_stationary_evaluator,
)
from .javascript import generate_javascript_radau_wrapper
from .wasm import WasmGenerator, dense_lu_solve_function


@dataclasses.dataclass(frozen=True)
class WasmLayout:
    """Public state-vector metadata required by a WebAssembly host."""

    state_size: int
    potentials: dict[str, int]
    currents: dict[str, dict[str, int]]


@dataclasses.dataclass(frozen=True)
class WasmArtifact:
    """A compiled WebAssembly solver and its matching JavaScript wrapper."""

    module: bytes
    javascript: str
    layout: WasmLayout


def compile_wasm(
    circuit: Circuit,
    *,
    ac_cases: Sequence[Mapping[str, Any]] | None = None,
) -> WasmArtifact:
    """Compile a circuit into a complete browser-oriented WebAssembly artifact.

    ``ac_cases`` is host metadata for independent small-signal inputs. Each
    mapping is copied into the generated wrapper and conventionally contains
    ``reference``, ``type``, and ``net`` fields.
    """
    system = compile_circuit(circuit)
    module = generate_wasm_solver(system)
    encoded_cases = [dict(case) for case in ac_cases] if ac_cases is not None else None
    javascript = generate_javascript_radau_wrapper(system, ac_cases=encoded_cases)
    currents: dict[str, dict[str, int]] = {}
    for (reference, port), index in system.layout.currents.items():
        currents.setdefault(reference, {})[port] = index
    layout = WasmLayout(len(system.state), dict(system.layout.potentials), currents)
    return WasmArtifact(module, javascript, layout)


def generate_wasm_solver(system: EquationSystem, *, memory_pages: int = 1) -> bytes:
    """Generate the solver module for an already flattened equation system."""
    if memory_pages < 1:
        msg = "memory_pages must be positive"
        raise ValueError(msg)
    step = discretize_radau_iia(system)
    functions = [
        transpile_radau_evaluator(step),
        transpile_stationary_evaluator(system),
    ]
    if system.auxiliaries:
        functions.append(transpile_auxiliary_evaluator(system))
    functions.append(transpile_ac_linearizer(system))
    if system.auxiliaries:
        functions.append(transpile_ac_auxiliary_linearizer(system))
    functions.append(dense_lu_solve_function())

    memory = radau_memory_layout(system)
    required_pages = max(1, (memory.byte_length + 65_535) // 65_536)
    return WasmGenerator(memory_pages=max(memory_pages, required_pages)).generate(tuple(functions))
