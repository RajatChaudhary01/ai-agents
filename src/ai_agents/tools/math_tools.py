"""A calculator tool that evaluates maths without ever running Python code.

The expression is parsed into an AST, checked against a small allow-list (numbers, the
arithmetic operators and functions/constants from :mod:`math`), and then evaluated by
walking the tree here. ``eval`` is never used.

Python integers have no size limit, so innocent-looking input such as ``9**9**9`` or
``factorial(10**6)`` could keep the CPU busy for minutes. Results are therefore capped at
``MAX_RESULT_DIGITS`` digits, and oversized work is refused *before* it is computed.
"""

import ast
import math
import operator
from collections.abc import Callable
from numbers import Number
from typing import Any

from smolagents import tool

MAX_EXPRESSION_LENGTH = 1_000
MAX_RESULT_DIGITS = 4_000  # below Python's 4300-digit int-to-str conversion limit
MAX_COMBINATORIC_ARG = 10_000  # largest argument accepted by factorial / comb / perm

BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Every public name in the math module (sqrt, sin, log, pi, e, factorial, ...) plus a few
# harmless builtins.
NAMES: dict[str, Any] = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}
NAMES |= {"abs": abs, "round": round, "min": min, "max": max}

_COMBINATORICS = frozenset({"factorial", "comb", "perm"})
_STRUCTURAL_NODES = (ast.Expression, ast.Load, *BINARY_OPERATORS, *UNARY_OPERATORS)


class UnsafeExpressionError(ValueError):
    """The expression uses something other than numbers, operators and math functions."""


class CalculationLimitError(ArithmeticError):
    """The calculation would exceed the configured resource limits."""


def check_expression(tree: ast.AST) -> None:
    """Raise ``UnsafeExpressionError`` unless every node of ``tree`` is on the allow-list."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, Number):
                raise UnsafeExpressionError(f"non-numeric constant {node.value!r}")
        elif isinstance(node, ast.Name):
            if node.id not in NAMES:
                raise UnsafeExpressionError(f"unknown name '{node.id}'")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpressionError("only plain function calls like sqrt(x) are allowed")
            if node.keywords:
                raise UnsafeExpressionError("keyword arguments are not supported")
        elif isinstance(node, ast.BinOp | ast.UnaryOp):
            continue  # operator types are checked as their own nodes below
        elif not isinstance(node, _STRUCTURAL_NODES):
            raise UnsafeExpressionError(f"'{type(node).__name__}' is not allowed")


def _digits(value: int) -> int:
    """Upper bound on the number of decimal digits of ``value`` (cheap, no conversion)."""
    return int(abs(value).bit_length() * 0.30103) + 1


def _limit(value: Any) -> Any:
    if isinstance(value, int) and _digits(value) > MAX_RESULT_DIGITS:
        raise CalculationLimitError(f"result has more than {MAX_RESULT_DIGITS} digits")
    return value


def _refuse_huge_integer_work(op: ast.operator, left: Any, right: Any) -> None:
    """Estimate integer ``**`` and ``*`` results before computing them."""
    if not (isinstance(left, int) and isinstance(right, int)):
        return  # floats overflow quickly with OverflowError instead
    too_big = False
    if isinstance(op, ast.Pow):
        too_big = right > 0 and abs(left) > 1 and right * math.log10(abs(left)) > MAX_RESULT_DIGITS
    elif isinstance(op, ast.Mult):
        too_big = _digits(left) + _digits(right) > MAX_RESULT_DIGITS + 1
    if too_big:
        raise CalculationLimitError(f"result has more than {MAX_RESULT_DIGITS} digits")


def evaluate(node: ast.AST) -> Any:
    """Evaluate a tree that has passed :func:`check_expression`."""
    match node:
        case ast.Expression(body=body):
            return evaluate(body)
        case ast.Constant(value=value):
            return value
        case ast.Name(id=name):
            return NAMES[name]
        case ast.UnaryOp(op=op, operand=operand):
            return UNARY_OPERATORS[type(op)](evaluate(operand))
        case ast.BinOp(left=left, op=op, right=right):
            a, b = evaluate(left), evaluate(right)
            _refuse_huge_integer_work(op, a, b)
            return _limit(BINARY_OPERATORS[type(op)](a, b))
        case ast.Call(func=ast.Name(id=name), args=args):
            values = [evaluate(arg) for arg in args]
            if name in _COMBINATORICS and any(
                isinstance(v, Number) and abs(v) > MAX_COMBINATORIC_ARG for v in values
            ):
                raise CalculationLimitError(
                    f"{name}() arguments must not exceed {MAX_COMBINATORIC_ARG}"
                )
            return _limit(NAMES[name](*values))
    raise UnsafeExpressionError(f"'{type(node).__name__}' is not allowed")


@tool
def calculate(expression: str) -> str:
    """Safely evaluates a maths expression and returns the result.

    Supports + - * / // % ** and parentheses, plus math functions and constants such as
    sqrt, sin, cos, tan, log, log10, exp, ceil, floor, factorial, radians, degrees, abs,
    round, min, max, pi and e. Integer results are limited to 4000 digits.

    Args:
        expression: The expression to evaluate, e.g. '125 * 4 + 2**5', 'sqrt(144) + 10' or 'sin(pi / 2)'.

    Returns:
        The result as a string, or a message starting with 'Error', 'Syntax error',
        'Security error', 'Math error' or 'Evaluation error' explaining what went wrong.
    """
    text = (expression or "").strip()
    if not text:
        return "Error: Empty expression, nothing to calculate."
    if len(text) > MAX_EXPRESSION_LENGTH:
        return (
            f"Error: Expression is too long ({len(text)} characters, max "
            f"{MAX_EXPRESSION_LENGTH}). Split it into smaller calculations."
        )

    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as exc:
        return f"Syntax error in '{text}': {exc}"
    except (RecursionError, MemoryError):
        # Very deep nesting makes the parser itself give up.
        return "Syntax error: the expression is nested too deeply."

    try:
        check_expression(tree)
    except UnsafeExpressionError as exc:
        return (
            f"Security error: {exc}. Only numbers, arithmetic operators and math "
            "functions are allowed."
        )

    try:
        return str(evaluate(tree))
    except ZeroDivisionError:
        return "Math error: Division by zero."
    except CalculationLimitError as exc:
        return (
            f"Math error: Calculation too large ({exc}). Use smaller numbers, or a float "
            "for an approximation, e.g. 2.0**1000."
        )
    except OverflowError:
        return "Math error: the result overflowed (too large for a float)."
    except RecursionError:
        return "Math error: the expression is nested too deeply."
    except UnsafeExpressionError as exc:
        return f"Security error: {exc}."
    except Exception as exc:  # noqa: BLE001 - e.g. sqrt(-1), factorial(2.5)
        return f"Evaluation error: {exc}"
