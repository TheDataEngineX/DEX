"""Tests for dataenginex._json helpers."""

from __future__ import annotations

from dataenginex import _json


class TestJsonHelpers:
    def test_dumps_basic(self) -> None:
        result = _json.dumps({"a": 1, "b": "hello"})
        assert '"a"' in result
        assert '"b"' in result
        assert "1" in result
        assert "hello" in result

    def test_dumps_indent(self) -> None:
        result = _json.dumps({"x": 1}, indent=2)
        assert "\n" in result

    def test_dumpb_returns_bytes(self) -> None:
        result = _json.dumpb({"a": 1})
        assert isinstance(result, bytes)
        assert b'"a"' in result

    def test_dumpb_indent(self) -> None:
        result = _json.dumpb({"x": 1}, indent=2)
        assert isinstance(result, bytes)
        assert b"\n" in result

    def test_loads_string(self) -> None:
        assert _json.loads('{"key": "value"}') == {"key": "value"}

    def test_loads_bytes(self) -> None:
        assert _json.loads(b'{"n": 42}') == {"n": 42}

    def test_roundtrip(self) -> None:
        data = {"nested": [1, 2, 3], "flag": True}
        assert _json.loads(_json.dumps(data)) == data

    def test_dumpb_roundtrip(self) -> None:
        data = {"nested": [1, 2, 3]}
        assert _json.loads(_json.dumpb(data)) == data
