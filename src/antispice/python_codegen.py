"""Generate native Python numerical kernels for an equation system."""

import dataclasses
from typing import Any

import wrenfold

from .compiler import (
    EquationSystem,
    discretize_radau_iia,
    transpile_ac_auxiliary_linearizer,
    transpile_ac_linearizer,
    transpile_auxiliary_evaluator,
    transpile_radau_evaluator,
    transpile_stationary_evaluator,
)


@dataclasses.dataclass(frozen=True)
class PythonKernels:
    """Circuit-specific native Python evaluator functions."""

    radau_evaluate: Any
    stationary_evaluate: Any
    evaluate_auxiliaries: Any | None
    ac_linearize: Any
    ac_auxiliary_linearize: Any | None
    source: str


def generate_python_kernels(system: EquationSystem, numpy: Any) -> PythonKernels:
    """Generate and compile all numerical evaluator functions."""
    step = discretize_radau_iia(system)
    definitions = [
        transpile_radau_evaluator(step),
        transpile_stationary_evaluator(system),
        transpile_ac_linearizer(system),
    ]
    if system.auxiliaries:
        definitions.extend(
            [
                transpile_auxiliary_evaluator(system),
                transpile_ac_auxiliary_linearizer(system),
            ]
        )
    generator = wrenfold.code_generation.PythonGenerator(use_output_arguments=True)
    source = "import numpy as np\n\n" + "\n\n".join(generator.generate(definition) for definition in definitions)
    namespace: dict[str, Any] = {"np": numpy}
    exec(compile(source, "<antispice-generated-python>", "exec"), namespace)
    return PythonKernels(
        radau_evaluate=namespace["radau_evaluate"],
        stationary_evaluate=namespace["stationary_evaluate"],
        evaluate_auxiliaries=namespace.get("evaluate_auxiliaries"),
        ac_linearize=namespace["ac_linearize"],
        ac_auxiliary_linearize=namespace.get("ac_auxiliary_linearize"),
        source=source,
    )
