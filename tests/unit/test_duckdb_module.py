"""Tests for v0.7 DuckDB portable SQL module."""


from dataenginex.duckdb import DuckDBCatalog, DuckDBEngine, PortableSQLValidator


class TestDuckDBEngine:
    def test_create_engine(self) -> None:
        engine = DuckDBEngine()
        assert engine.database == ":memory:"

    def test_engine_lifecycle(self) -> None:
        engine = DuckDBEngine()
        assert engine.is_open() is False
        engine.open()
        assert engine.is_open() is True
        engine.close()
        assert engine.is_open() is False

    def test_execute(self) -> None:
        engine = DuckDBEngine()
        engine.open()
        result = engine.execute("SELECT 1")
        assert result["status"] == "executed"


class TestDuckDBCatalog:
    def test_register_schema(self) -> None:
        catalog = DuckDBCatalog()
        catalog.register_schema("main", ["users", "orders"])
        schemas = catalog.list_schemas()
        assert "main" in schemas

    def test_list_tables(self) -> None:
        catalog = DuckDBCatalog()
        catalog.register_schema("main", ["users", "orders"])
        tables = catalog.list_tables("main")
        assert "users" in tables

    def test_table_to_resource(self) -> None:
        catalog = DuckDBCatalog()
        resource = catalog.table_to_resource("main", "users")
        assert resource["name"] == "users"
        assert resource["provider"] == "duckdb"


class TestPortableSQLValidator:
    def test_validate_portable(self) -> None:
        validator = PortableSQLValidator()
        result = validator.validate("SELECT 1")
        assert result["is_portable"] is True

    def test_check_compatibility(self) -> None:
        validator = PortableSQLValidator()
        result = validator.check_compatibility("SELECT 1", "spark")
        assert result["compatible"] is True
