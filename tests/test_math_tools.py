import math
import time

import pytest

from ai_agents.tools import calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("125 * 4 + 2**5", 532),
        ("sqrt(144) + 10", 22.0),
        ("(1 + 2) * 3", 9),
        ("17 // 5", 3),
        ("17 % 5", 2),
        ("-3 + +5", 2),
        ("max(1, 7, 3)", 7),
        ("round(2.567, 2)", 2.57),
    ],
)
def test_valid_expressions(expression, expected):
    assert float(calculate(expression=expression)) == pytest.approx(expected)


def test_math_constants():
    assert float(calculate(expression="sin(pi / 2)")) == pytest.approx(1.0)
    assert float(calculate(expression="e")) == pytest.approx(math.e)


@pytest.mark.parametrize("expression", ["", "   "])
def test_empty_expression(expression):
    assert calculate(expression=expression).startswith("Error: Empty expression")


def test_syntax_error():
    assert calculate(expression="2 +* 3").startswith("Syntax error")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "math.sqrt(4)",
        "[x for x in range(3)]",
        "lambda: 1",
        "x + 1",
    ],
)
def test_unsafe_expressions_are_rejected(expression):
    assert calculate(expression=expression).startswith("Security error")


def test_division_by_zero():
    assert "Division by zero" in calculate(expression="1 / 0")


def test_overflow():
    assert "overflow" in calculate(expression="10.0 ** 10000")


def test_evaluation_error():
    assert calculate(expression="sqrt(-1)").startswith("Evaluation error")


@pytest.mark.parametrize(
    "expression",
    [
        "10**10**8",
        "9**9**9",
        "(-2)**10**7",
        "factorial(10**6)",
        "comb(10**6, 500000)",
        "perm(20000, 10000)",
        "factorial(3000) * factorial(3000)",
        "2**14000",
        "factorial(5000)",
    ],
)
def test_huge_calculations_are_rejected_quickly(expression):
    start = time.perf_counter()
    result = calculate(expression=expression)
    assert result.startswith("Math error: Calculation too large"), result
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    ("expression", "expected_digits"),
    [("2**13000", 3914), ("factorial(1000)", 2568), ("10**3999", 4000)],
)
def test_large_but_allowed_integers(expression, expected_digits):
    assert len(calculate(expression=expression)) == expected_digits


def test_float_power_overflow_still_reported():
    assert "overflow" in calculate(expression="2.0 ** 100000")


@pytest.mark.parametrize("expression", ["'a' * 10**9", "b'x' * 3", "'abc'"])
def test_non_numeric_constants_are_rejected(expression):
    assert calculate(expression=expression).startswith("Security error")


@pytest.mark.parametrize(
    "expression",
    [
        "-" * 200_000 + "1",  # ast.parse raises MemoryError
        "1+" * 50_000 + "1",  # ast.parse raises RecursionError
        "1" * 1_001,
    ],
)
def test_huge_or_deeply_nested_expression_returns_error(expression):
    result = calculate(expression=expression)
    assert result.startswith(("Error", "Syntax error"))


def test_null_byte_is_a_syntax_error():
    assert calculate(expression="1\x00+1").startswith("Syntax error")
