"""Tests for AI tools builtin registration."""

from __future__ import annotations

from typing import Any

from dataenginex.domains.ai.tools.builtin import register_builtin_tools


class TestRegisterBuiltinTools:
    def test_register_no_args(self) -> None:
        register_builtin_tools()

    def test_register_with_lakehouse(self, tmp_path: Any) -> None:
        register_builtin_tools(lakehouse_dir=tmp_path)

    def test_register_with_models_dir(self, tmp_path: Any) -> None:
        register_builtin_tools(models_dir=tmp_path)

    def test_register_with_vector_store(self) -> None:
        register_builtin_tools(vector_store=object())
