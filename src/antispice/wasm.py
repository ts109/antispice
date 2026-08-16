"""Direct WebAssembly binary generation for Wrenfold functions.

The ABI uses ``f64``/``i64``/``i32`` for scalar float/integer/boolean inputs.
Matrices and all output arguments are byte offsets into the exported linear
memory and use row-major storage.  Scalar return values use the WebAssembly
function result directly.
"""

import dataclasses
import json
import math
import struct
import typing

import wrenfold
from wrenfold import ast, type_info

_I32 = 0x7F
_I64 = 0x7E
_F64 = 0x7C
_EMPTY_BLOCK = 0x40


@dataclasses.dataclass(frozen=True)
class WasmArgument:
    """Description of one argument in the generated WebAssembly ABI."""

    name: str
    wasm_type: str
    direction: str
    rows: int | None = None
    cols: int | None = None


@dataclasses.dataclass(frozen=True)
class WasmFunction:
    """ABI metadata for one exported function."""

    name: str
    arguments: tuple[WasmArgument, ...]
    return_type: str | None


class WasmGenerator(wrenfold.BaseGenerator):
    """Generate a complete WebAssembly 1.0 binary from Wrenfold's AST.

    Functions involving non-core mathematical operations import them from
    ``math_module`` using lowercase names such as ``sin``, ``exp``, and
    ``powf``.  Memory is exported as ``memory``.
    """

    def __init__(self, *, memory_pages: int = 1, math_module: str = "math") -> None:
        super().__init__()
        if memory_pages < 1:
            raise ValueError("memory_pages must be positive")
        self.memory_pages = memory_pages
        self.math_module = math_module
        self.abi: tuple[WasmFunction, ...] = ()

    def generate(
        self,
        definition: ast.FunctionDefinition | typing.Sequence[ast.FunctionDefinition],
    ) -> bytes:
        """Generate a directly executable WebAssembly module binary."""
        definitions = tuple(definition) if isinstance(definition, typing.Sequence) else (definition,)
        if not definitions:
            raise ValueError("at least one function definition is required")

        imports = _collect_math_imports(definitions)
        import_indices = {name: index for index, name in enumerate(imports)}

        type_entries: list[bytes] = []
        type_indices: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}

        def intern_type(parameters: tuple[int, ...], results: tuple[int, ...]) -> int:
            key = (parameters, results)
            if key not in type_indices:
                type_indices[key] = len(type_entries)
                type_entries.append(b"\x60" + _vector(bytes([value]) for value in parameters) + _vector(bytes([value]) for value in results))
            return type_indices[key]

        import_type_indices = {name: intern_type(*_math_import_signature(name)) for name in imports}
        function_type_indices: list[int] = []
        abi: list[WasmFunction] = []
        for function in definitions:
            parameters, results, function_abi = _function_signature(function.signature)
            function_type_indices.append(intern_type(parameters, results))
            abi.append(function_abi)
        self.abi = tuple(abi)

        sections = [_section(1, _vector(type_entries))]
        if imports:
            sections.append(
                _section(
                    2,
                    _vector(_name(self.math_module) + _name(name) + b"\x00" + _unsigned(import_type_indices[name]) for name in imports),
                )
            )
        sections.extend(
            [
                _section(3, _vector(_unsigned(index) for index in function_type_indices)),
                _section(5, _vector((b"\x00" + _unsigned(self.memory_pages),))),
            ]
        )

        export_entries = [
            _name("memory") + b"\x02" + _unsigned(0),
            *(_name(function.signature.name) + b"\x00" + _unsigned(len(imports) + index) for index, function in enumerate(definitions)),
        ]
        sections.append(_section(7, _vector(export_entries)))

        bodies = []
        for function in definitions:
            emitter = _FunctionEmitter(
                function,
                import_indices=import_indices,
            )
            bodies.append(emitter.emit())
        sections.append(_section(10, _vector(bodies)))

        metadata = json.dumps(
            [dataclasses.asdict(function) for function in self.abi],
            separators=(",", ":"),
        ).encode()
        sections.append(_section(0, _name("antispice.abi") + metadata))
        return b"\x00asm\x01\x00\x00\x00" + b"".join(sections)


class _FunctionEmitter:
    def __init__(
        self,
        function: ast.FunctionDefinition,
        *,
        import_indices: dict[str, int],
    ) -> None:
        self.function = function
        self.import_indices = import_indices
        self.local_indices: dict[str, int] = {}
        self.local_types: dict[str, int] = {}
        self.argument_indices: dict[str, int] = {}
        self.argument_types: dict[str, int] = {}

        for index, argument in enumerate(function.signature.arguments):
            wasm_type = _argument_wasm_type(argument)
            self.argument_indices[argument.name] = index
            self.argument_types[argument.name] = wasm_type

        next_index = len(function.signature.arguments)
        self.locals: list[int] = []
        for statement in _walk_statements(function.body):
            if isinstance(statement, ast.Declaration):
                wasm_type = _numeric_wasm_type(statement.type)
                self.local_indices[statement.name] = next_index
                self.local_types[statement.name] = wasm_type
                self.locals.append(wasm_type)
                next_index += 1

    def emit(self) -> bytes:
        local_declarations = _group_locals(self.locals)
        code = bytearray()
        for statement in self.function.body:
            code.extend(self._statement(statement))
        code.append(0x0B)
        body = local_declarations + bytes(code)
        return _unsigned(len(body)) + body

    def _statement(self, statement: typing.Any) -> bytes:
        if isinstance(statement, ast.Comment):
            return b""
        if isinstance(statement, ast.Declaration):
            if statement.value is None:
                return b""
            return self._expression(statement.value) + _local_set(self.local_indices[statement.name])
        if isinstance(statement, ast.AssignTemporary):
            return self._expression(statement.right) + _local_set(self.local_indices[statement.left])
        if isinstance(statement, ast.Branch):
            return self._expression(statement.condition) + bytes([0x04, _EMPTY_BLOCK]) + self._span(statement.if_branch) + b"\x05" + self._span(statement.else_branch) + b"\x0b"
        if isinstance(statement, ast.OptionalOutputBranch):
            pointer = self.argument_indices[statement.argument.name]
            return _local_get(pointer) + b"\x41\x00\x47" + bytes([0x04, _EMPTY_BLOCK]) + self._span(statement.statements) + b"\x0b"
        if isinstance(statement, ast.AssignOutputScalar):
            pointer = self.argument_indices[statement.arg.name]
            value_type = _numeric_wasm_type(statement.arg.type)
            return _local_get(pointer) + self._expression(statement.value) + _store(value_type, 0)
        if isinstance(statement, ast.AssignOutputMatrix):
            pointer = self.argument_indices[statement.arg.name]
            values = tuple(statement.value.args)
            return b"".join(_local_get(pointer) + self._expression(value) + _store(_F64, index * 8) for index, value in enumerate(values))
        if isinstance(statement, ast.ReturnObject):
            return self._expression(statement.value) + b"\x0f"
        raise NotImplementedError(f"unsupported Wrenfold statement: {type(statement).__name__}")

    def _span(self, span: ast.AstSpan) -> bytes:
        return b"".join(self._statement(statement) for statement in span)

    def _expression(self, expression: typing.Any) -> bytes:
        if isinstance(expression, ast.VariableRef):
            return _local_get(self.local_indices[expression.name])
        if isinstance(expression, ast.GetArgument):
            return _local_get(self.argument_indices[expression.argument.name])
        if isinstance(expression, ast.GetMatrixElement):
            if not isinstance(expression.arg, ast.GetArgument):
                raise NotImplementedError("matrix temporaries are not supported")
            argument = expression.arg.argument
            offset = (expression.row * argument.type.cols + expression.col) * 8
            return _local_get(self.argument_indices[argument.name]) + _load(_F64, offset)
        if isinstance(expression, ast.IntegerLiteral):
            return b"\x42" + _signed(expression.value)
        if isinstance(expression, ast.FloatLiteral):
            return b"\x44" + struct.pack("<d", expression.value)
        if isinstance(expression, ast.BooleanLiteral):
            return b"\x41" + _signed(int(expression.value))
        if isinstance(expression, ast.SpecialConstant):
            value = math.e if expression.value == wrenfold.SymbolicConstant.Euler else math.pi
            return b"\x44" + struct.pack("<d", value)
        if isinstance(expression, (ast.Parenthetical,)):
            return self._expression(expression.contents)
        if isinstance(expression, ast.Add):
            return self._fold(expression.args, expression, _add_opcode)
        if isinstance(expression, ast.Multiply):
            return self._fold(expression.args, expression, _multiply_opcode)
        if isinstance(expression, ast.Divide):
            value_type = self._expression_type(expression.left)
            opcode = 0xA3 if value_type == _F64 else 0x7F
            return self._expression(expression.left) + self._expression(expression.right) + bytes([opcode])
        if isinstance(expression, ast.Negate):
            value_type = self._expression_type(expression.arg)
            if value_type == _F64:
                return self._expression(expression.arg) + b"\x9a"
            return b"\x42\x00" + self._expression(expression.arg) + b"\x7d"
        if isinstance(expression, ast.Cast):
            return self._cast(expression)
        if isinstance(expression, ast.Compare):
            return self._compare(expression)
        if isinstance(expression, ast.CallStdFunction):
            return self._call_standard_function(expression)
        if isinstance(expression, ast.Ternary):
            return self._expression(expression.left) + self._expression(expression.right) + self._expression(expression.condition) + b"\x1b"
        raise NotImplementedError(f"unsupported Wrenfold expression: {type(expression).__name__}")

    def _fold(
        self,
        arguments: ast.AstSpan,
        expression: typing.Any,
        opcode_for_type: typing.Callable[[int], int],
    ) -> bytes:
        values = tuple(arguments)
        if not values:
            raise ValueError("an arithmetic expression must have operands")
        opcode = bytes([opcode_for_type(self._expression_type(expression))])
        result = bytearray(self._expression(values[0]))
        for value in values[1:]:
            result.extend(self._expression(value))
            result.extend(opcode)
        return bytes(result)

    def _cast(self, expression: ast.Cast) -> bytes:
        source_type = self._expression_type(expression.arg)
        destination_type = _numeric_type_to_wasm(expression.destination_type)
        code = self._expression(expression.arg)
        if source_type == destination_type:
            return code
        conversions = {
            (_I32, _I64): b"\xac",
            (_I32, _F64): b"\xb7",
            (_I64, _I32): b"\xa7",
            (_I64, _F64): b"\xb9",
            (_F64, _I32): b"\xaa",
            (_F64, _I64): b"\xb0",
        }
        try:
            return code + conversions[source_type, destination_type]
        except KeyError as error:
            raise NotImplementedError(f"unsupported cast from {source_type:#x} to {destination_type:#x}") from error

    def _compare(self, expression: ast.Compare) -> bytes:
        value_type = self._expression_type(expression.left)
        operations = {
            (_F64, wrenfold.RelationalOperation.LessThan): 0x63,
            (_F64, wrenfold.RelationalOperation.LessThanOrEqual): 0x65,
            (_F64, wrenfold.RelationalOperation.Equal): 0x61,
            (_I64, wrenfold.RelationalOperation.LessThan): 0x53,
            (_I64, wrenfold.RelationalOperation.LessThanOrEqual): 0x57,
            (_I64, wrenfold.RelationalOperation.Equal): 0x51,
            (_I32, wrenfold.RelationalOperation.LessThan): 0x48,
            (_I32, wrenfold.RelationalOperation.LessThanOrEqual): 0x4C,
            (_I32, wrenfold.RelationalOperation.Equal): 0x46,
        }
        try:
            opcode = operations[value_type, expression.operation]
        except KeyError as error:
            raise NotImplementedError(f"unsupported comparison: {expression.operation}") from error
        return self._expression(expression.left) + self._expression(expression.right) + bytes([opcode])

    def _call_standard_function(self, expression: ast.CallStdFunction) -> bytes:
        function = expression.function
        arguments = tuple(expression.args)
        if function == wrenfold.StdMathFunction.Abs:
            return self._expression(arguments[0]) + b"\x99"
        if function == wrenfold.StdMathFunction.Sqrt:
            return self._expression(arguments[0]) + b"\x9f"
        if function == wrenfold.StdMathFunction.Floor:
            return self._expression(arguments[0]) + b"\x9c\xb0"
        if function == wrenfold.StdMathFunction.Signum:
            argument = arguments[0]
            return self._expression(argument) + b"\x44" + struct.pack("<d", 0.0) + b"\x64\xad" + self._expression(argument) + b"\x44" + struct.pack("<d", 0.0) + b"\x63\xad\x7d"
        name = _math_function_name(function)
        return b"".join(self._expression(argument) for argument in arguments) + b"\x10" + _unsigned(self.import_indices[name])

    def _expression_type(self, expression: typing.Any) -> int:
        if isinstance(expression, ast.VariableRef):
            return self.local_types[expression.name]
        if isinstance(expression, ast.GetArgument):
            return self.argument_types[expression.argument.name]
        if isinstance(expression, ast.GetMatrixElement):
            return _F64
        if isinstance(expression, ast.IntegerLiteral):
            return _I64
        if isinstance(expression, ast.FloatLiteral | ast.SpecialConstant):
            return _F64
        if isinstance(expression, ast.BooleanLiteral | ast.Compare):
            return _I32
        if isinstance(expression, ast.Cast):
            return _numeric_type_to_wasm(expression.destination_type)
        if isinstance(expression, ast.CallStdFunction):
            if expression.function in (
                wrenfold.StdMathFunction.Floor,
                wrenfold.StdMathFunction.Signum,
            ):
                return _I64
            return _F64
        if isinstance(expression, ast.Ternary):
            return self._expression_type(expression.left)
        if isinstance(expression, ast.Parenthetical):
            return self._expression_type(expression.contents)
        if isinstance(expression, (ast.Add, ast.Multiply)):
            return self._expression_type(next(iter(expression.args)))
        if isinstance(expression, ast.Divide):
            return self._expression_type(expression.left)
        if isinstance(expression, ast.Negate):
            return self._expression_type(expression.arg)
        raise NotImplementedError(f"cannot determine type of {type(expression).__name__}")


def generate_wasm(
    description: wrenfold.FunctionDescription,
    *,
    memory_pages: int = 1,
    math_module: str = "math",
    convert_ternaries: bool = True,
) -> bytes:
    """Transpile a Wrenfold function description and generate WebAssembly."""
    definition = wrenfold.code_generation.transpile(
        description,
        convert_ternaries=convert_ternaries,
    )
    return WasmGenerator(
        memory_pages=memory_pages,
        math_module=math_module,
    ).generate(definition)


def _function_signature(
    signature: ast.FunctionSignature,
) -> tuple[tuple[int, ...], tuple[int, ...], WasmFunction]:
    parameters = tuple(_argument_wasm_type(argument) for argument in signature.arguments)
    result = () if signature.return_type is None else (_numeric_wasm_type(signature.return_type),)
    arguments = tuple(
        WasmArgument(
            name=argument.name,
            wasm_type=_wasm_type_name(_argument_wasm_type(argument)),
            direction={
                wrenfold.code_generation.ArgumentDirection.Input: "input",
                wrenfold.code_generation.ArgumentDirection.Output: "output",
                wrenfold.code_generation.ArgumentDirection.OptionalOutput: "optional_output",
            }[argument.direction],
            rows=argument.type.rows if isinstance(argument.type, type_info.MatrixType) else None,
            cols=argument.type.cols if isinstance(argument.type, type_info.MatrixType) else None,
        )
        for argument in signature.arguments
    )
    return (
        parameters,
        result,
        WasmFunction(
            name=signature.name,
            arguments=arguments,
            return_type=None if not result else _wasm_type_name(result[0]),
        ),
    )


def _argument_wasm_type(argument: wrenfold.code_generation.Argument) -> int:
    if isinstance(argument.type, type_info.MatrixType) or not argument.is_input:
        return _I32
    return _numeric_wasm_type(argument.type)


def _numeric_wasm_type(value_type: typing.Any) -> int:
    if not isinstance(value_type, type_info.ScalarType):
        raise NotImplementedError(f"unsupported Wrenfold type: {value_type}")
    return _numeric_type_to_wasm(value_type.numeric_type)


def _numeric_type_to_wasm(numeric_type: type_info.NumericType) -> int:
    return {
        type_info.NumericType.Bool: _I32,
        type_info.NumericType.Integer: _I64,
        type_info.NumericType.Float: _F64,
    }[numeric_type]


def _wasm_type_name(value_type: int) -> str:
    return {_I32: "i32", _I64: "i64", _F64: "f64"}[value_type]


def _walk_statements(span: ast.AstSpan) -> typing.Iterator[typing.Any]:
    for statement in span:
        yield statement
        if isinstance(statement, ast.Branch):
            yield from _walk_statements(statement.if_branch)
            yield from _walk_statements(statement.else_branch)
        elif isinstance(statement, ast.OptionalOutputBranch):
            yield from _walk_statements(statement.statements)


def _walk_expressions(value: typing.Any) -> typing.Iterator[typing.Any]:
    if type(value).__module__.startswith("pywrenfold.ast"):
        yield value
    if isinstance(value, ast.AstSpan):
        for child in value:
            yield from _walk_expressions(child)
        return
    for attribute in (
        "value",
        "condition",
        "left",
        "right",
        "arg",
        "args",
        "if_branch",
        "else_branch",
        "body",
        "statements",
    ):
        if hasattr(value, attribute):
            child = getattr(value, attribute)
            if type(child).__module__.startswith("pywrenfold.ast"):
                yield from _walk_expressions(child)


def _collect_math_imports(
    definitions: tuple[ast.FunctionDefinition, ...],
) -> tuple[str, ...]:
    imports: dict[str, None] = {}
    inline = {
        wrenfold.StdMathFunction.Abs,
        wrenfold.StdMathFunction.Sqrt,
        wrenfold.StdMathFunction.Floor,
        wrenfold.StdMathFunction.Signum,
    }
    for definition in definitions:
        for node in _walk_expressions(definition.body):
            if isinstance(node, ast.CallStdFunction) and node.function not in inline:
                imports.setdefault(_math_function_name(node.function), None)
    return tuple(imports)


def _math_function_name(function: wrenfold.StdMathFunction) -> str:
    return function.name.lower()


def _math_import_signature(name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if name in {"atan2", "powf"}:
        return (_F64, _F64), (_F64,)
    if name == "powi":
        return (_F64, _I64), (_F64,)
    return (_F64,), (_F64,)


def _group_locals(locals_: list[int]) -> bytes:
    groups: list[tuple[int, int]] = []
    for value_type in locals_:
        if groups and groups[-1][1] == value_type:
            groups[-1] = groups[-1][0] + 1, value_type
        else:
            groups.append((1, value_type))
    return _vector(_unsigned(count) + bytes([value_type]) for count, value_type in groups)


def _add_opcode(value_type: int) -> int:
    return {_I32: 0x6A, _I64: 0x7C, _F64: 0xA0}[value_type]


def _multiply_opcode(value_type: int) -> int:
    return {_I32: 0x6C, _I64: 0x7E, _F64: 0xA2}[value_type]


def _local_get(index: int) -> bytes:
    return b"\x20" + _unsigned(index)


def _local_set(index: int) -> bytes:
    return b"\x21" + _unsigned(index)


def _load(value_type: int, offset: int) -> bytes:
    opcode, alignment = {
        _I32: (0x28, 2),
        _I64: (0x29, 3),
        _F64: (0x2B, 3),
    }[value_type]
    return bytes([opcode]) + _unsigned(alignment) + _unsigned(offset)


def _store(value_type: int, offset: int) -> bytes:
    opcode, alignment = {
        _I32: (0x36, 2),
        _I64: (0x37, 3),
        _F64: (0x39, 3),
    }[value_type]
    return bytes([opcode]) + _unsigned(alignment) + _unsigned(offset)


def _unsigned(value: int) -> bytes:
    if value < 0:
        raise ValueError("unsigned LEB128 cannot encode a negative value")
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        encoded.append(byte)
        if not value:
            return bytes(encoded)


def _signed(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        sign_bit = byte & 0x40
        done = (value == 0 and not sign_bit) or (value == -1 and sign_bit)
        encoded.append(byte if done else byte | 0x80)
        if done:
            return bytes(encoded)


def _name(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _unsigned(len(encoded)) + encoded


def _vector(values: typing.Iterable[bytes]) -> bytes:
    values = tuple(values)
    return _unsigned(len(values)) + b"".join(values)


def _section(section_id: int, contents: bytes) -> bytes:
    return bytes([section_id]) + _unsigned(len(contents)) + contents
