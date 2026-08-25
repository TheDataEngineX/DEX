"""Tests for AI tools helpers: _echo, _referenced_table_names, _require_read_only."""

from __future__ import annotations

import pytest

from dataenginex.domains.ai.tools.builtin import (
    _echo,
    _list_tools,
    _referenced_table_names,
    _require_read_only,
)


class TestEcho:
    def test_returns_message(self) -> None:
        assert _echo("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert _echo("") == ""

    def test_special_chars(self) -> None:
        assert _echo("a@b#c$") == "a@b#c$"


class TestListTools:
    def test_returns_list(self) -> None:
        tools = _list_tools()
        assert isinstance(tools, list)


class TestRequireReadOnly:
    def test_select_ok(self) -> None:
        _require_read_only("SELECT 1")

    def test_with_ok(self) -> None:
        _require_read_only("WITH cte AS (SELECT 1) SELECT * FROM cte")

    def test_select_with_semicolon(self) -> None:
        _require_read_only("SELECT 1;")

    def test_insert_rejected(self) -> None:
        with pytest.raises(ValueError, match="read-only"):
            _require_read_only("INSERT INTO t VALUES (1)")

    def test_delete_rejected(self) -> None:
        with pytest.raises(ValueError, match="read-only"):
            _require_read_only("DELETE FROM t")

    def test_multiple_statements_rejected(self) -> None:
        with pytest.raises(ValueError, match="single statement"):
            _require_read_only("SELECT 1; SELECT 2")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="read-only"):
            _require_read_only("")

    def test_with_leading_parens(self) -> None:
        _require_read_only("(SELECT 1)")


class TestReferencedTableNames:
    def test_from_clause(self) -> None:
        result = _referenced_table_names("SELECT * FROM users")
        assert result == {"users"}

    def test_join_clause(self) -> None:
        result = _referenced_table_names(
            "SELECT * FROM orders JOIN users ON orders.user_id = users.id"
        )
        assert "orders" in result
        assert "users" in result

    def test_quoted_names(self) -> None:
        result = _referenced_table_names('SELECT * FROM "my_table"')
        assert result == {"my_table"}

    def test_no_tables(self) -> None:
        result = _referenced_table_names("SELECT 1")
        assert result == set()

    def test_multiple_from(self) -> None:
        # Regex matches FROM <name>, so "FROM a, b" only captures "a"
        result = _referenced_table_names("SELECT * FROM a, b")
        assert result == {"a"}
