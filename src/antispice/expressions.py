"""Compile safe circuit-expression syntax into Wrenfold expressions."""

import ast
import math
import typing

import wrenfold

from .circuit import Expression

type SymbolicExpression = wrenfold.sym.Expr


class UnknownName(ValueError):
    """An expression references a name not present in its environment."""


def parse_expression(
    source: str,
    environment: dict[str, Expression | SymbolicExpression],
    *,
    context: str,
) -> SymbolicExpression:
    """Parse one non-executing arithmetic expression."""
    try:
        tree = ast.parse(source, mode="eval")
        return _ExpressionCompiler(environment).visit(tree.body)
    except UnknownName:
        raise
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        msg = f"invalid {context}: {source!r}: {error}"
        raise ValueError(msg) from error


class _ExpressionCompiler(ast.NodeVisitor):
    """Small, non-executing expression compiler for circuit equations."""

    _binary_operators: typing.ClassVar = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
    }
    _comparisons: typing.ClassVar = {
        ast.Lt: lambda left, right: left < right,
        ast.LtE: lambda left, right: left <= right,
        ast.Gt: lambda left, right: left > right,
        ast.GtE: lambda left, right: left >= right,
    }
    _functions: typing.ClassVar = {name: getattr(wrenfold.sym, name) for name in ("sin", "cos", "tan", "exp", "log", "sqrt", "where")}

    def __init__(self, environment: dict[str, Expression | SymbolicExpression]) -> None:
        self.environment = {"pi": math.pi, "e": math.e, **environment}

    def visit_Constant(self, node: ast.Constant) -> typing.Any:
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            msg = "only numeric constants are allowed"
            raise TypeError(msg)
        return node.value

    def visit_Name(self, node: ast.Name) -> typing.Any:
        try:
            return self.environment[node.id]
        except KeyError as error:
            msg = f"unknown name {node.id!r}"
            raise UnknownName(msg) from error

    def visit_BinOp(self, node: ast.BinOp) -> typing.Any:
        try:
            operation = self._binary_operators[type(node.op)]
        except KeyError as error:
            msg = f"unsupported operator: {type(node.op).__name__}"
            raise ValueError(msg) from error
        return operation(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> typing.Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        msg = f"unsupported unary operator: {type(node.op).__name__}"
        raise ValueError(msg)

    def visit_Compare(self, node: ast.Compare) -> typing.Any:
        if len(node.ops) != 1:
            msg = "chained comparisons are not supported"
            raise ValueError(msg)
        try:
            operation = self._comparisons[type(node.ops[0])]
        except KeyError as error:
            msg = f"unsupported comparison: {type(node.ops[0]).__name__}"
            raise ValueError(msg) from error
        return operation(self.visit(node.left), self.visit(node.comparators[0]))

    def visit_Call(self, node: ast.Call) -> typing.Any:
        if not isinstance(node.func, ast.Name) or node.keywords:
            msg = "only simple function calls are allowed"
            raise ValueError(msg)
        if node.func.id == "abs":
            if len(node.args) != 1:
                msg = "abs expects one argument"
                raise ValueError(msg)
            return abs(self.visit(node.args[0]))
        try:
            function = self._functions[node.func.id]
        except KeyError as error:
            msg = f"unknown function {node.func.id!r}"
            raise ValueError(msg) from error
        return function(*(self.visit(argument) for argument in node.args))

    def generic_visit(self, node: ast.AST) -> typing.NoReturn:
        msg = f"unsupported expression syntax: {type(node).__name__}"
        raise ValueError(msg)
