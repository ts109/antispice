"""Generate a high-level JavaScript wrapper for a WebAssembly Radau solver."""

import json
import re
from importlib.resources import files
from typing import Any

from .compiler import EquationSystem, differential_state_indices, radau_memory_layout


def generate_javascript_radau_wrapper(
    system: EquationSystem,
    *,
    class_name: str = "AntispiceSolver",
    function_name: str = "radau_evaluate",
    ac_cases: list[dict[str, Any]] | None = None,
) -> str:
    """Generate an ES module wrapping the flattened WebAssembly solver ABI."""
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", class_name) is None:
        msg = f"invalid JavaScript class name: {class_name!r}"
        raise ValueError(msg)
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", function_name) is None:
        msg = f"invalid WebAssembly function name: {function_name!r}"
        raise ValueError(msg)

    memory = radau_memory_layout(system)
    currents: dict[str, dict[str, int]] = {}
    for (reference, port), index in system.layout.currents.items():
        currents.setdefault(reference, {})[port] = index

    source = files("antispice").joinpath("templates/radau_solver.js").read_text(encoding="utf-8")
    replacements = {
        "__CLASS_NAME__": class_name,
        "__FUNCTION_NAME__": json.dumps(function_name),
        "__STATE_SIZE__": str(len(system.state)),
        "__DIFFERENTIAL_STATE_INDICES__": json.dumps(differential_state_indices(system)),
        "__POTENTIALS__": json.dumps(system.layout.potentials, ensure_ascii=False),
        "__CURRENTS__": json.dumps(currents, ensure_ascii=False),
        "__AUXILIARIES__": json.dumps(
            {
                reference: {name: index for index, (element, name) in enumerate(system.auxiliaries) if element == reference}
                for reference in dict.fromkeys(element for element, _ in system.auxiliaries)
            },
            ensure_ascii=False,
        ),
        "__AUXILIARY_COUNT__": str(len(system.auxiliaries)),
        "__AC_CASES__": json.dumps(ac_cases or (), ensure_ascii=False),
        "__STAGE_DERIVATIVES__": str(memory.stage_derivatives),
        "__PREVIOUS_STATE__": str(memory.previous_state),
        "__UPDATED_STAGE_DERIVATIVES__": str(memory.updated_stage_derivatives),
        "__NEXT_STATE__": str(memory.next_state),
        "__RESIDUAL__": str(memory.residual),
        "__JACOBIAN__": str(memory.jacobian),
        "__AUXILIARY_VALUES__": str(memory.auxiliary_values),
        "__AC_STATE_JACOBIAN__": str(memory.ac_state_jacobian),
        "__AC_DERIVATIVE_JACOBIAN__": str(memory.ac_derivative_jacobian),
        "__AC_AUXILIARY_STATE_JACOBIAN__": str(memory.ac_auxiliary_state_jacobian),
        "__AC_AUXILIARY_DERIVATIVE_JACOBIAN__": str(memory.ac_auxiliary_derivative_jacobian),
        "__AC_MATRIX__": str(memory.ac_matrix),
        "__AC_RHS__": str(memory.ac_rhs),
    }
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    return source
