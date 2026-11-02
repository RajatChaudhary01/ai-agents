import ast
import math
import operator
from collections.abc import Callable
from numbers import Number
from typing import Any

from smolagents import tool

# Whitelist safe functions and constants from the standard math library
SAFE_MATH: dict[str, object] = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}
SAFE_MATH.update(
    {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pi": math.pi,
        "e": math.e,
    }
)

# Allowed AST node types for safe mathematical evaluation
ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)

# Resource limits. Python ints never overflow, so without these an expression such as
# 10**10**8 or factorial(10**6) would keep the CPU busy for minutes and block the agent.
MAX_RESULT_DIGITS = 4000  # stays below Python's 4300-digit int->str conversion limit
MAX_COMBINATORIC_ARG = 10_000  # cap for factorial / comb / perm arguments
_COMBINATORIC_FUNCS = frozenset({"factorial", "comb", "perm"})

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculationLimitError(ArithmeticError):
    """Raised when a calculation would exceed the configured resource limits."""


def _validate_ast(node: ast.AST) -> tuple[bool, str]:
    """Ensures expression AST contains only safe mathematical operations."""
    for child in ast.walk(node):
        if not isinstance(child, ALLOWED_NODES):
            return False, f"Disallowed syntax element: {type(child).__name__}"
        if isinstance(child, ast.Constant) and not isinstance(child.value, Number):
            return False, f"Disallowed non-numeric constant: {child.value!r}"
        if isinstance(child, ast.Name) and child.id not in SAFE_MATH:
            return False, f"Disallowed variable or identifier: '{child.id}'"
        if isinstance(child, ast.Call):  # noqa: SIM102
            if not isinstance(child.func, ast.Name) or child.func.id not in SAFE_MATH:
                func_name = child.func.id if isinstance(child.func, ast.Name) else "unknown"
                return False, f"Disallowed function call: '{func_name}'"
    return True, ""


def _int_digits(value: int) -> int:
    """Upper-bound estimate of the number of decimal digits in ``value``."""
    return int(abs(value).bit_length() * math.log10(2)) + 1


def _check_result_size(value: Any) -> Any:
    if isinstance(value, int) and _int_digits(value) > MAX_RESULT_DIGITS:
        raise CalculationLimitError(f"result has more than {MAX_RESULT_DIGITS} digits")
    return value


def _check_binop(op: ast.operator, left: Any, right: Any) -> None:
    """Reject integer operations whose result would be too large, before computing them."""
    if not (isinstance(left, int) and isinstance(right, int)):
        return  # float operations overflow quickly with OverflowError instead
    if isinstance(op, ast.Pow) and right > 0 and abs(left) > 1:
        if right * math.log10(abs(left)) > MAX_RESULT_DIGITS:
            raise CalculationLimitError(f"result has more than {MAX_RESULT_DIGITS} digits")
    elif (
        isinstance(op, ast.Mult) and _int_digits(left) + _int_digits(right) > MAX_RESULT_DIGITS + 1
    ):
        raise CalculationLimitError(f"result has more than {MAX_RESULT_DIGITS} digits")


def _evaluate(node: ast.AST) -> Any:
    """Evaluate a validated expression AST without using ``eval``."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return SAFE_MATH[node.id]
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        _check_binop(node.op, left, right)
        return _check_result_size(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.Call):
        name = node.func.id  # type: ignore[attr-defined]  # guaranteed by _validate_ast
        args = [_evaluate(arg) for arg in node.args]
        if name in _COMBINATORIC_FUNCS and any(
            isinstance(a, Number) and abs(a) > MAX_COMBINATORIC_ARG for a in args
        ):
            raise CalculationLimitError(
                f"{name}() arguments must not exceed {MAX_COMBINATORIC_ARG}"
            )
        return _check_result_size(SAFE_MATH[name](*args))  # type: ignore[operator]
    raise ValueError(f"Unsupported syntax element: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """A safe mathematical calculator tool that evaluates arithmetic and mathematical expressions.

    Supports arithmetic operations (+, -, *, /, //, %, **), parentheses, and math functions
    such as sqrt, sin, cos, tan, log, log10, ceil, floor, radians, degrees, abs, round, min, max, pi, and e.
    Integer results are limited to 4000 digits.

    Args:
        expression: The mathematical expression string to evaluate (e.g. '125 * 4 + 2**5', 'sqrt(144) + 10', 'sin(pi / 2)').

    Returns:
        The evaluated numerical result as a string, or an error message if the expression is invalid.
    """
    cleaned = expression.strip()
    if not cleaned:
        return "Error: Empty expression provided."

    try:
        parsed_tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        return f"Syntax error in mathematical expression '{cleaned}': {e}"

    is_safe, error_msg = _validate_ast(parsed_tree)
    if not is_safe:
        return f"Security error: {error_msg}. Only basic arithmetic and standard math functions are allowed."

    try:
        return str(_evaluate(parsed_tree))
    except ZeroDivisionError:
        return "Math error: Division by zero is undefined."
    except CalculationLimitError as e:
        return f"Math error: Calculation too large ({e}). Use smaller numbers, or floats for an approximation, e.g. 2.0**1000."
    except OverflowError:
        return "Math error: Numerical calculation resulted in an overflow."
    except Exception as e:  # noqa: BLE001
        return f"Evaluation error: {e}"
