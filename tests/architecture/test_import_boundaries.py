"""Architecture tests enforcing import boundaries (§5.2, ADR-0001).

Foundation must not import PySpark, FastAPI, cloud SDKs, ML frameworks, or provider
implementations. Studio must not import private runtime/spark modules.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "dataenginex"

# Modules that foundation must NEVER import
FOUNDATION_FORBIDDEN = {
    "pyspark", "spark", "fastapi", "uvicorn", "starlette",
    "boto3", "google.cloud", "google.auth",
    "torch", "tensorflow", "sklearn", "xgboost",
    "mlflow", "sentence_transformers",
    "qdrant_client",
    "confluent_kafka", "pika",
    "strawberry",
    "deltalake",
}

# Foundation source directories
FOUNDATION_DIRS = [
    "foundation",
]


def _imported_modules(tree: ast.Module) -> set[str]:
    """Extract top-level module names from an AST."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _python_files(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory.rglob("*.py"))


@pytest.mark.parametrize(
    "src_dir",
    FOUNDATION_DIRS,
    ids=lambda d: f"foundation/{d}",
)
def test_foundation_has_no_forbidden_imports(src_dir: str) -> None:
    """Foundation modules must not import framework or provider dependencies."""
    target = SRC / src_dir
    if not target.exists():
        pytest.skip(f"{target} does not exist")

    violations: list[str] = []
    for py_file in _python_files(target):
        if py_file.name == "__init__.py" and py_file.stat().st_size == 0:
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue

        imported = _imported_modules(tree)
        found = imported & FOUNDATION_FORBIDDEN
        if found:
            violations.append(f"{py_file.relative_to(SRC)}: imports {found}")

    assert not violations, "Foundation has forbidden imports:\n" + "\n".join(violations)


def test_public_api_exposes_gateway_protocol() -> None:
    """The public surface must expose DexGateway for third-party clients."""
    gateway_file = SRC / "interfaces" / "gateway.py"
    assert gateway_file.exists(), "interfaces/gateway.py missing"
    content = gateway_file.read_text()
    assert "class DexGateway" in content, "DexGateway protocol not found in public surface"


def test_foundation_no_sqlalchemy_imports() -> None:
    """Foundation must not depend on SQLAlchemy (persistence is a provider concern)."""
    target = SRC / "foundation"
    if not target.exists():
        pytest.skip("foundation/ does not exist")

    violations: list[str] = []
    for py_file in _python_files(target):
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        imported = _imported_modules(tree)
        if "sqlalchemy" in imported:
            violations.append(f"{py_file.relative_to(SRC)}: imports sqlalchemy")

    assert not violations, "Foundation must not import SQLAlchemy:\n" + "\n".join(violations)
