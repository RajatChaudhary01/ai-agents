import math

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
