"""Compile a :mod:`antispice.circuit` into a symbolic Radau-IIA solver.

The public circuit model is first flattened into the canonical differential
algebraic equation system ``F(t, x, xdot) = 0``.  Numerical methods consume that
representation and do not need to know about models, parts, ports, or nets.
"""

import dataclasses
import typing

import wrenfold

from .circuit import Circuit, Expression
from .expressions import UnknownName as _UnknownName
from .expressions import parse_expression as _parse_expression

REFERENCE_NODE = "0"

type SymbolicExpression = wrenfold.sym.Expr
type Expressions = tuple[SymbolicExpression, ...]
type ElementPort = tuple[str, str]


@dataclasses.dataclass(frozen=True)
class StateLayout:
    """Semantic indices of the flat DAE state vector."""

    potentials: dict[str, int]
    currents: dict[ElementPort, int]

    def potential_index(self, node: str) -> int:
        """Return the state-vector index of a non-reference potential."""
        if node == REFERENCE_NODE:
            msg = "the reference node has no state-vector potential"
            raise ValueError(msg)

        return self.potentials[node]

    def current_index(self, element: str, port: str) -> int:
        """Return the state-vector index of an element port current."""
        return self.currents[element, port]


@dataclasses.dataclass(frozen=True)
class EquationSystem:
    """Canonical symbolic circuit DAE ``F(t, x, xdot) = 0``."""

    time: SymbolicExpression
    state: Expressions
    state_derivative: Expressions
    equations: Expressions
    layout: StateLayout
    auxiliaries: dict[tuple[str, str], SymbolicExpression] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        size = len(self.state)

        if len(self.state_derivative) != size:
            msg = "state and state derivative must have equal size"
            raise ValueError(msg)

        if len(self.equations) != size:
            msg = f"equation system is not square: {len(self.equations)} equations for {size} state variables"
            raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class RadauStepSystem:
    """Two-stage, third-order Radau-IIA discretization of an equation system."""

    time_start: SymbolicExpression
    step_size: SymbolicExpression
    previous_state: Expressions
    stage_derivatives: Expressions
    next_state: Expressions
    equations: Expressions
    layout: StateLayout


@dataclasses.dataclass(frozen=True)
class RadauMemoryLayout:
    """Byte offsets of the vectors used by the WebAssembly Radau ABI."""

    stage_derivatives: int
    previous_state: int
    updated_stage_derivatives: int
    next_state: int
    residual: int
    jacobian: int
    auxiliary_values: int
    ac_state_jacobian: int
    ac_derivative_jacobian: int
    ac_auxiliary_state_jacobian: int
    ac_auxiliary_derivative_jacobian: int
    ac_matrix: int
    ac_rhs: int
    byte_length: int


def radau_memory_layout(system: EquationSystem) -> RadauMemoryLayout:
    """Return a compact, eight-byte-aligned WebAssembly memory layout."""
    state_size = len(system.state)
    stage_derivatives = 0
    previous_state = stage_derivatives + 2 * state_size * 8
    updated_stage_derivatives = previous_state + state_size * 8
    next_state = updated_stage_derivatives + 2 * state_size * 8
    residual = next_state + state_size * 8
    jacobian = residual + 2 * state_size * 8
    auxiliary_values = jacobian + (2 * state_size) ** 2 * 8
    ac_state_jacobian = auxiliary_values + len(system.auxiliaries) * 8
    ac_derivative_jacobian = ac_state_jacobian + state_size**2 * 8
    ac_auxiliary_state_jacobian = ac_derivative_jacobian + state_size**2 * 8
    ac_auxiliary_derivative_jacobian = ac_auxiliary_state_jacobian + len(system.auxiliaries) * state_size * 8
    ac_matrix = ac_auxiliary_derivative_jacobian + len(system.auxiliaries) * state_size * 8
    ac_dimension = 2 * (state_size + 1)
    ac_rhs = ac_matrix + ac_dimension**2 * 8
    return RadauMemoryLayout(
        stage_derivatives=stage_derivatives,
        previous_state=previous_state,
        updated_stage_derivatives=updated_stage_derivatives,
        next_state=next_state,
        residual=residual,
        jacobian=jacobian,
        auxiliary_values=auxiliary_values,
        ac_state_jacobian=ac_state_jacobian,
        ac_derivative_jacobian=ac_derivative_jacobian,
        ac_auxiliary_state_jacobian=ac_auxiliary_state_jacobian,
        ac_auxiliary_derivative_jacobian=ac_auxiliary_derivative_jacobian,
        ac_matrix=ac_matrix,
        ac_rhs=ac_rhs,
        byte_length=ac_rhs + ac_dimension * 8,
    )


def compile_circuit(circuit: Circuit) -> EquationSystem:
    """Resolve and compile a circuit into its canonical symbolic DAE."""
    circuit.validate()

    nodes = _ordered_nodes(circuit)
    potentials = {node: wrenfold.sym.symbol(f"Phi_{node}") for node in nodes if node != REFERENCE_NODE}
    potential_derivatives = {node: wrenfold.sym.symbol(f"Phidot_{node}") for node in nodes if node != REFERENCE_NODE}

    currents: dict[ElementPort, SymbolicExpression] = {}
    current_derivatives: dict[ElementPort, SymbolicExpression] = {}
    for element_name, element in circuit.elements.items():
        model = circuit.resolve_model(element)
        for port in model.ports[1:]:
            key = (element_name, port)
            currents[key] = wrenfold.sym.symbol(f"I_{element_name}_{port}")
            current_derivatives[key] = wrenfold.sym.symbol(f"Idot_{element_name}_{port}")

    state = (*potentials.values(), *currents.values())
    state_derivative = (
        *potential_derivatives.values(),
        *current_derivatives.values(),
    )
    layout = StateLayout(
        potentials={node: index for index, node in enumerate(potentials)},
        currents={key: len(potentials) + index for index, key in enumerate(currents)},
    )

    time = wrenfold.sym.symbol("t")
    kcl = dict.fromkeys(potentials, wrenfold.sym.zero)
    constitutive_equations: list[SymbolicExpression] = []
    auxiliary_expressions: dict[tuple[str, str], SymbolicExpression] = {}

    for element_name, element in circuit.elements.items():
        model = circuit.resolve_model(element)
        parameters = circuit.resolve_parameters(element)
        port_nodes = dict(zip(model.ports, element.nodes, strict=True))
        reference_port = model.ports[0]
        reference_net = port_nodes[reference_port]

        environment: dict[str, Expression | SymbolicExpression] = {"t": time}
        environment.update(_compile_parameters(parameters, time=time, element_name=element_name))

        reference_current = wrenfold.sym.zero
        for port in model.ports[1:]:
            node = port_nodes[port]
            current = currents[element_name, port]
            current_derivative = current_derivatives[element_name, port]

            environment[f"U_{port}"] = _potential(potentials, node) - _potential(potentials, reference_net)
            environment[f"Udot_{port}"] = _potential(potential_derivatives, node) - _potential(potential_derivatives, reference_net)
            environment[f"I_{port}"] = current
            environment[f"Idot_{port}"] = current_derivative

            if node != REFERENCE_NODE:
                kcl[node] += current
            reference_current -= current

        if reference_net != REFERENCE_NODE:
            kcl[reference_net] += reference_current

        compiled_auxiliaries = _compile_auxiliaries(
            model.auxiliaries,
            environment=environment,
            element_name=element_name,
        )
        environment.update(compiled_auxiliaries)
        auxiliary_expressions.update(((element_name, name), expression) for name, expression in compiled_auxiliaries.items())

        for equation in model.equations:
            constitutive_equations.append(
                _parse_expression(
                    equation,
                    environment,
                    context=f"equation of element {element_name!r}",
                )
            )

    return EquationSystem(
        time=time,
        state=state,
        state_derivative=state_derivative,
        equations=(*kcl.values(), *constitutive_equations),
        layout=layout,
        auxiliaries=auxiliary_expressions,
    )


def discretize_radau_iia(system: EquationSystem) -> RadauStepSystem:
    """Apply the two-stage, third-order Radau-IIA method to a DAE."""
    size = len(system.state)
    previous_state = _unique_symbols(size)
    derivative_a = _unique_symbols(size)
    derivative_b = _unique_symbols(size)
    stage_derivatives = (*derivative_a, *derivative_b)
    time_start, step_size = wrenfold.sym.make_symbols(["t0", "h"])

    x0 = _column(previous_state)
    ka = _column(derivative_a)
    kb = _column(derivative_b)
    state_a = x0 + step_size * (5 / 12 * ka - 1 / 12 * kb)
    state_b = x0 + step_size * (3 / 4 * ka + 1 / 4 * kb)

    equations_a = _substitute_system(
        system,
        state=state_a,
        derivative=ka,
        time=time_start + step_size / 3,
    )
    equations_b = _substitute_system(
        system,
        state=state_b,
        derivative=kb,
        time=time_start + step_size,
    )

    return RadauStepSystem(
        time_start=time_start,
        step_size=step_size,
        previous_state=previous_state,
        stage_derivatives=stage_derivatives,
        next_state=tuple(state_b.to_flat_list()),
        equations=(*equations_a, *equations_b),
        layout=system.layout,
    )


def transpile_radau_evaluator(step: RadauStepSystem, *, function_name: str = "radau_evaluate") -> typing.Any:
    """Create Wrenfold IR that evaluates a Radau residual and Jacobian."""
    size = len(step.stage_derivatives)
    residual = _column(step.equations)
    jacobian = wrenfold.sym.jacobian(step.equations, step.stage_derivatives)
    description = wrenfold.FunctionDescription(function_name)
    input_derivatives = description.add_input_argument(
        "stage_derivatives",
        wrenfold.type_info.MatrixType(rows=size, cols=1),
    )
    input_time = description.add_input_argument(
        "t0",
        wrenfold.type_info.ScalarType(wrenfold.type_info.NumericType.Float),
    )
    input_step = description.add_input_argument(
        "h",
        wrenfold.type_info.ScalarType(wrenfold.type_info.NumericType.Float),
    )
    input_state = description.add_input_argument(
        "previous_state",
        wrenfold.type_info.MatrixType(rows=len(step.previous_state), cols=1),
    )
    replacements = [
        *zip(step.stage_derivatives, input_derivatives, strict=True),
        *zip(step.previous_state, input_state, strict=True),
        (step.time_start, input_time),
        (step.step_size, input_step),
    ]
    description.add_output_argument("residual", is_optional=False, value=residual.subs(replacements))
    description.add_output_argument("jacobian", is_optional=False, value=jacobian.subs(replacements))
    return wrenfold.code_generation.transpile(description)


def transpile_stationary_evaluator(system: EquationSystem, *, function_name: str = "stationary_evaluate") -> typing.Any:
    """Create Wrenfold IR for ``F(t, x, 0)`` and its state Jacobian."""
    residual = _column(system.equations).subs(list(zip(system.state_derivative, [0] * len(system.state_derivative), strict=True)))
    jacobian = wrenfold.sym.jacobian(residual, system.state)
    description = wrenfold.FunctionDescription(function_name)
    input_state = description.add_input_argument(
        "state",
        wrenfold.type_info.MatrixType(rows=len(system.state), cols=1),
    )
    input_time = description.add_input_argument(
        "time",
        wrenfold.type_info.ScalarType(wrenfold.type_info.NumericType.Float),
    )
    replacements = [
        *zip(system.state, input_state, strict=True),
        (system.time, input_time),
    ]
    description.add_output_argument("residual", is_optional=False, value=residual.subs(replacements))
    description.add_output_argument("jacobian", is_optional=False, value=jacobian.subs(replacements))
    return wrenfold.code_generation.transpile(description)


def transpile_auxiliary_evaluator(system: EquationSystem, *, function_name: str = "evaluate_auxiliaries") -> typing.Any:
    """Create Wrenfold IR evaluating all named model auxiliaries."""
    values = _column(tuple(system.auxiliaries.values()))
    description = wrenfold.FunctionDescription(function_name)
    input_state = description.add_input_argument("state", wrenfold.type_info.MatrixType(rows=len(system.state), cols=1))
    input_derivative = description.add_input_argument(
        "state_derivative",
        wrenfold.type_info.MatrixType(rows=len(system.state_derivative), cols=1),
    )
    input_time = description.add_input_argument("time", wrenfold.type_info.ScalarType(wrenfold.type_info.NumericType.Float))
    replacements = [
        *zip(system.state, input_state, strict=True),
        *zip(system.state_derivative, input_derivative, strict=True),
        (system.time, input_time),
    ]
    description.add_output_argument("values", is_optional=False, value=values.subs(replacements))
    return wrenfold.code_generation.transpile(description)


def transpile_ac_linearizer(system: EquationSystem, *, function_name: str = "ac_linearize") -> typing.Any:
    """Create Wrenfold IR for the stationary DAE state and derivative Jacobians."""
    derivative_zeros = list(zip(system.state_derivative, [0] * len(system.state_derivative), strict=True))
    state_jacobian = wrenfold.sym.jacobian(system.equations, system.state).subs(derivative_zeros)
    derivative_jacobian = wrenfold.sym.jacobian(system.equations, system.state_derivative).subs(derivative_zeros)
    description = wrenfold.FunctionDescription(function_name)
    input_state = description.add_input_argument("state", wrenfold.type_info.MatrixType(rows=len(system.state), cols=1))
    input_time = description.add_input_argument("time", wrenfold.type_info.ScalarType(wrenfold.type_info.NumericType.Float))
    replacements = [*zip(system.state, input_state, strict=True), (system.time, input_time)]
    description.add_output_argument("state_jacobian", is_optional=False, value=state_jacobian.subs(replacements))
    description.add_output_argument("derivative_jacobian", is_optional=False, value=derivative_jacobian.subs(replacements))
    return wrenfold.code_generation.transpile(description)


def transpile_ac_auxiliary_linearizer(system: EquationSystem, *, function_name: str = "ac_auxiliary_linearize") -> typing.Any:
    """Create Wrenfold IR for named auxiliary small-signal output Jacobians."""
    values = tuple(system.auxiliaries.values())
    derivative_zeros = list(zip(system.state_derivative, [0] * len(system.state_derivative), strict=True))
    state_jacobian = wrenfold.sym.jacobian(values, system.state).subs(derivative_zeros)
    derivative_jacobian = wrenfold.sym.jacobian(values, system.state_derivative).subs(derivative_zeros)
    description = wrenfold.FunctionDescription(function_name)
    input_state = description.add_input_argument("state", wrenfold.type_info.MatrixType(rows=len(system.state), cols=1))
    input_time = description.add_input_argument("time", wrenfold.type_info.ScalarType(wrenfold.type_info.NumericType.Float))
    replacements = [*zip(system.state, input_state, strict=True), (system.time, input_time)]
    description.add_output_argument("state_jacobian", is_optional=False, value=state_jacobian.subs(replacements))
    description.add_output_argument("derivative_jacobian", is_optional=False, value=derivative_jacobian.subs(replacements))
    return wrenfold.code_generation.transpile(description)


def _ordered_nodes(circuit: Circuit) -> tuple[str, ...]:
    nodes: dict[str, None] = {}

    for element in circuit.elements.values():
        for node in element.nodes:
            nodes.setdefault(node, None)

    if REFERENCE_NODE not in nodes:
        msg = f"circuit must contain reference node {REFERENCE_NODE!r}"
        raise ValueError(msg)

    return tuple(nodes)


def _potential(potentials: dict[str, SymbolicExpression], node: str) -> SymbolicExpression | int:
    if node == REFERENCE_NODE:
        return 0

    return potentials[node]


def _compile_parameters(
    parameters: dict[str, Expression],
    *,
    time: SymbolicExpression,
    element_name: str,
) -> dict[str, Expression | SymbolicExpression]:
    compiled: dict[str, Expression | SymbolicExpression] = {}
    pending = dict(parameters)

    while pending:
        progressed = False

        for name, value in tuple(pending.items()):
            if not isinstance(value, str):
                compiled[name] = value
                del pending[name]
                progressed = True

                continue

            try:
                compiled[name] = _parse_expression(
                    value,
                    {"t": time, **compiled},
                    context=f"parameter {name!r} of element {element_name!r}",
                )
            except _UnknownName:
                continue

            del pending[name]

            progressed = True

        if not progressed:
            names = ", ".join(sorted(pending))
            msg = f"unresolved or cyclic parameters on element {element_name!r}: {names}"
            raise ValueError(msg)

    return compiled


def _compile_auxiliaries(
    auxiliaries: dict[str, str],
    *,
    environment: dict[str, Expression | SymbolicExpression],
    element_name: str,
) -> dict[str, SymbolicExpression]:
    """Compile model-local expressions, resolving dependencies by substitution."""
    collisions = auxiliaries.keys() & environment.keys()
    if collisions:
        names = ", ".join(sorted(collisions))
        msg = f"auxiliary symbols collide with existing names on element {element_name!r}: {names}"
        raise ValueError(msg)

    compiled: dict[str, SymbolicExpression] = {}
    pending = dict(auxiliaries)
    while pending:
        progressed = False
        for name, expression in tuple(pending.items()):
            try:
                compiled[name] = _parse_expression(
                    expression,
                    {**environment, **compiled},
                    context=f"auxiliary symbol {name!r} of element {element_name!r}",
                )
            except _UnknownName:
                continue
            del pending[name]
            progressed = True

        if not progressed:
            names = ", ".join(sorted(pending))
            msg = f"unresolved or cyclic auxiliary symbols on element {element_name!r}: {names}"
            raise ValueError(msg)

    return compiled


def _substitute_system(
    system: EquationSystem,
    *,
    state: wrenfold.sym.MatrixExpr,
    derivative: wrenfold.sym.MatrixExpr,
    time: SymbolicExpression,
) -> Expressions:
    replacements = [
        *zip(system.state, state, strict=True),
        *zip(system.state_derivative, derivative, strict=True),
        (system.time, time),
    ]
    return tuple(wrenfold.sym.subs(equation, replacements) for equation in system.equations)


def _column(expressions: typing.Iterable[SymbolicExpression]) -> wrenfold.sym.MatrixExpr:
    return wrenfold.sym.matrix(tuple(expressions))


def _unique_symbols(count: int) -> Expressions:
    symbols = wrenfold.sym.unique_symbols(count=count)
    if count == 1:
        return (symbols,)
    return tuple(symbols)
