"""Generate a high-level JavaScript wrapper for a WebAssembly Radau solver."""

import json
import re
from importlib.resources import files

from .compiler import EquationSystem, radau_memory_layout


def generate_javascript_radau_wrapper(
    system: EquationSystem,
    *,
    class_name: str = "AntispiceSolver",
    function_name: str = "radau_evaluate",
) -> str:
    """Generate an ES module wrapping the flattened WebAssembly solver ABI."""
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", class_name) is None:
        raise ValueError(f"invalid JavaScript class name: {class_name!r}")
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", function_name) is None:
        raise ValueError(f"invalid WebAssembly function name: {function_name!r}")

    memory = radau_memory_layout(system)
    currents: dict[str, dict[str, int]] = {}
    for (reference, port), index in system.layout.currents.items():
        currents.setdefault(reference, {})[port] = index

    source = files("antispice").joinpath("templates/radau_solver.js").read_text(encoding="utf-8")
    replacements = {
        "__CLASS_NAME__": class_name,
        "__FUNCTION_NAME__": json.dumps(function_name),
        "__STATE_SIZE__": str(len(system.state)),
        "__POTENTIALS__": json.dumps(system.layout.potentials, ensure_ascii=False),
        "__CURRENTS__": json.dumps(currents, ensure_ascii=False),
        "__STAGE_DERIVATIVES__": str(memory.stage_derivatives),
        "__PREVIOUS_STATE__": str(memory.previous_state),
        "__UPDATED_STAGE_DERIVATIVES__": str(memory.updated_stage_derivatives),
        "__NEXT_STATE__": str(memory.next_state),
        "__RESIDUAL__": str(memory.residual),
        "__JACOBIAN__": str(memory.jacobian),
    }
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    return source
