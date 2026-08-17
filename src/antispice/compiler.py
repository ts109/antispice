"""Compile a :mod:`antispice.circuit` into a symbolic Radau-IIA solver.

The public circuit model is first flattened into the canonical differential
algebraic equation system ``F(t, x, xdot) = 0``.  Numerical methods consume that
representation and do not need to know about models, parts, ports, or nets.
"""

import ast
import dataclasses
import math
import typing

import wrenfold

from .circuit import Circuit, Expression

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
        if node == REFERENCE_NODE:
            raise ValueError("the reference node has no state-vector potential")

        return self.potentials[node]

    def current_index(self, element: str, port: str) -> int:
        return self.currents[element, port]


@dataclasses.dataclass(frozen=True)
class EquationSystem:
    """Canonical symbolic circuit DAE ``F(t, x, xdot) = 0``."""

    time: SymbolicExpression
    state: Expressions
    state_derivative: Expressions
    equations: Expressions
    layout: StateLayout

    def __post_init__(self) -> None:
        size = len(self.state)

        if len(self.state_derivative) != size:
            raise ValueError("state and state derivative must have equal size")

        if len(self.equations) != size:
            raise ValueError(f"equation system is not square: {len(self.equations)} equations for {size} state variables")


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
    return RadauMemoryLayout(
        stage_derivatives=stage_derivatives,
        previous_state=previous_state,
        updated_stage_derivatives=updated_stage_derivatives,
        next_state=next_state,
        residual=residual,
        jacobian=jacobian,
        byte_length=jacobian + (2 * state_size) ** 2 * 8,
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

        environment.update(
            _compile_auxiliaries(
                model.auxiliaries,
                environment=environment,
                element_name=element_name,
            )
        )

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


def transpile_radau_newton_step(step: RadauStepSystem, *, function_name: str = "radau_newton_step") -> typing.Any:
    """Create Wrenfold IR for one Newton update of one Radau timestep.

    The generated function accepts the current stage-derivative estimate,
    ``t0``, ``h``, and the previous state.  It returns an improved estimate and
    the corresponding end-of-step state.  Apply it repeatedly until converged.
    """
    residual = _column(step.equations)
    unknowns = _column(step.stage_derivatives)
    jacobian = wrenfold.sym.jacobian(step.equations, step.stage_derivatives)
    updated = unknowns - _solve_lu(jacobian, residual)

    size = len(step.previous_state)
    derivative_a = updated[:size, :]
    derivative_b = updated[size:, :]
    next_state = _column(step.previous_state) + step.step_size * (3 / 4 * derivative_a + 1 / 4 * derivative_b)

    description = wrenfold.FunctionDescription(function_name)
    input_derivatives = description.add_input_argument(
        "stage_derivatives",
        wrenfold.type_info.MatrixType(rows=2 * size, cols=1),
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
        wrenfold.type_info.MatrixType(rows=size, cols=1),
    )

    replacements = [
        *zip(step.stage_derivatives, input_derivatives, strict=True),
        *zip(step.previous_state, input_state, strict=True),
        (step.time_start, input_time),
        (step.step_size, input_step),
    ]
    description.add_output_argument(
        "updated_stage_derivatives",
        is_optional=False,
        value=updated.subs(replacements),
    )
    description.add_output_argument(
        "next_state",
        is_optional=False,
        value=next_state.subs(replacements),
    )
    return wrenfold.code_generation.transpile(description)


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


def generate_python_radau_solver(
    system: EquationSystem,
    *,
    function_name: str = "radau_newton_step",
) -> str:
    """Generate Python source for the Radau-IIA Newton update function."""
    step = discretize_radau_iia(system)
    function = transpile_radau_newton_step(step, function_name=function_name)
    source = wrenfold.code_generation.PythonGenerator().generate(function)
    return f"import numpy as np\n\n{source}"


def generate_wasm_radau_solver(
    system: EquationSystem,
    *,
    function_name: str = "radau_evaluate",
    memory_pages: int = 1,
) -> bytes:
    """Generate a WebAssembly binary containing the Radau Newton update."""
    from .wasm import WasmGenerator, dense_lu_solve_function

    if memory_pages < 1:
        raise ValueError("memory_pages must be positive")
    required_pages = (radau_memory_layout(system).byte_length + 65_535) // 65_536
    step = discretize_radau_iia(system)
    evaluator = transpile_radau_evaluator(step, function_name=function_name)
    stationary = transpile_stationary_evaluator(system)
    solver = dense_lu_solve_function()
    return WasmGenerator(memory_pages=max(memory_pages, required_pages)).generate((evaluator, stationary, solver))


def _ordered_nodes(circuit: Circuit) -> tuple[str, ...]:
    nodes: dict[str, None] = {}

    for element in circuit.elements.values():
        for node in element.nodes:
            nodes.setdefault(node, None)

    if REFERENCE_NODE not in nodes:
        raise ValueError(f"circuit must contain reference node {REFERENCE_NODE!r}")

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
            raise ValueError(f"unresolved or cyclic parameters on element {element_name!r}: {names}")

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
        raise ValueError(f"auxiliary symbols collide with existing names on element {element_name!r}: {names}")

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
            raise ValueError(f"unresolved or cyclic auxiliary symbols on element {element_name!r}: {names}")

    return compiled


class _UnknownName(ValueError):
    pass


def _parse_expression(
    source: str,
    environment: dict[str, Expression | SymbolicExpression],
    *,
    context: str,
) -> SymbolicExpression:
    try:
        tree = ast.parse(source, mode="eval")
        return _ExpressionCompiler(environment).visit(tree.body)
    except _UnknownName:
        raise
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid {context}: {source!r}: {error}") from error


class _ExpressionCompiler(ast.NodeVisitor):
    """Small, non-executing expression compiler for circuit equations."""

    _binary_operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
    }
    _comparisons = {
        ast.Lt: lambda left, right: left < right,
        ast.LtE: lambda left, right: left <= right,
        ast.Gt: lambda left, right: left > right,
        ast.GtE: lambda left, right: left >= right,
    }
    _functions = {name: getattr(wrenfold.sym, name) for name in ("sin", "cos", "tan", "exp", "log", "sqrt", "where")}

    def __init__(self, environment: dict[str, Expression | SymbolicExpression]) -> None:
        self.environment = {"pi": math.pi, "e": math.e, **environment}

    def visit_Constant(self, node: ast.Constant) -> typing.Any:
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("only numeric constants are allowed")

        return node.value

    def visit_Name(self, node: ast.Name) -> typing.Any:
        try:
            return self.environment[node.id]
        except KeyError as error:
            raise _UnknownName(f"unknown name {node.id!r}") from error

    def visit_BinOp(self, node: ast.BinOp) -> typing.Any:
        try:
            operation = self._binary_operators[type(node.op)]
        except KeyError as error:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}") from error

        return operation(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> typing.Any:
        operand = self.visit(node.operand)

        if isinstance(node.op, ast.UAdd):
            return operand

        if isinstance(node.op, ast.USub):
            return -operand

        raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> typing.Any:
        if len(node.ops) != 1:
            raise ValueError("chained comparisons are not supported")
        try:
            operation = self._comparisons[type(node.ops[0])]
        except KeyError as error:
            raise ValueError(f"unsupported comparison: {type(node.ops[0]).__name__}") from error

        return operation(self.visit(node.left), self.visit(node.comparators[0]))

    def visit_Call(self, node: ast.Call) -> typing.Any:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("only simple function calls are allowed")

        if node.func.id == "abs":
            if len(node.args) != 1:
                raise ValueError("abs expects one argument")

            return abs(self.visit(node.args[0]))

        try:
            function = self._functions[node.func.id]
        except KeyError as error:
            raise ValueError(f"unknown function {node.func.id!r}") from error

        return function(*(self.visit(argument) for argument in node.args))

    def generic_visit(self, node: ast.AST) -> typing.NoReturn:
        raise ValueError(f"unsupported expression syntax: {type(node).__name__}")


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


def _solve_lower(matrix: wrenfold.sym.MatrixExpr, right_hand_side: wrenfold.sym.MatrixExpr) -> wrenfold.sym.MatrixExpr:
    rows, columns = matrix.shape

    if rows != columns or right_hand_side.shape[0] != rows:
        raise ValueError("incompatible lower-triangular system")

    result = [[None] * right_hand_side.shape[1] for _ in range(rows)]

    for column in range(right_hand_side.shape[1]):
        for row in range(rows):
            value = right_hand_side[row, column]
            for inner in range(row):
                value -= matrix[row, inner] * result[inner][column]
            result[row][column] = value / matrix[row, row]

    return wrenfold.sym.matrix(result)


def _solve_upper(matrix: wrenfold.sym.MatrixExpr, right_hand_side: wrenfold.sym.MatrixExpr) -> wrenfold.sym.MatrixExpr:
    rows, columns = matrix.shape

    if rows != columns or right_hand_side.shape[0] != rows:
        raise ValueError("incompatible upper-triangular system")

    result = [[None] * right_hand_side.shape[1] for _ in range(rows)]

    for column in range(right_hand_side.shape[1]):
        for row in reversed(range(rows)):
            value = right_hand_side[row, column]
            for inner in range(row + 1, rows):
                value -= matrix[row, inner] * result[inner][column]
            result[row][column] = value / matrix[row, row]

    return wrenfold.sym.matrix(result)


def _solve_lu(matrix: wrenfold.sym.MatrixExpr, right_hand_side: wrenfold.sym.MatrixExpr) -> wrenfold.sym.MatrixExpr:
    permutation_rows, lower, upper, permutation_columns = wrenfold.sym.full_piv_lu(matrix)
    intermediate = _solve_lower(lower, permutation_rows.T * right_hand_side)
    solution = _solve_upper(upper, intermediate)

    return permutation_columns.T * solution
