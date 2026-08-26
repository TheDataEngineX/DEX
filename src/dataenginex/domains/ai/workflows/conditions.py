"""Conditional branching for agent workflows."""

from __future__ import annotations

from typing import Any


class Condition:
    """Evaluates a condition against a context dict.

    Supported operators: eq, ne, gt, lt, contains.
    """

    OPERATORS = {"eq", "ne", "gt", "lt", "contains"}

    def __init__(self, field: str, operator: str, value: Any) -> None:
        if operator not in self.OPERATORS:
            msg = f"Unknown operator: {operator!r}. Must be one of {self.OPERATORS}"
            raise ValueError(msg)
        self.field = field
        self.operator = operator
        self.value = value

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate the condition against the given context.

        Args:
            context: Dict containing the field to evaluate.

        Returns:
            True if the condition is met.

        Raises:
            KeyError: If the field is not in the context.
        """
        actual = context[self.field]
        if self.operator == "eq":
            return bool(actual == self.value)
        if self.operator == "ne":
            return bool(actual != self.value)
        if self.operator == "gt":
            return bool(actual > self.value)
        if self.operator == "lt":
            return bool(actual < self.value)
        if self.operator == "contains":
            return bool(self.value in actual)
        msg = f"Unhandled operator: {self.operator!r}"
        raise ValueError(msg)
