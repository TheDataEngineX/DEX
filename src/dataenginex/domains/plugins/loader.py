"""Project-local plugin loader.

Mirrors the installed-package plugin discovery in
dataenginex.plugins.registry.discover(), but scans project_dir/plugins/*.py
directly so a project can define its own connectors, transforms, etc.
without publishing an installable package. Modules register themselves via
the same decorators core connectors/transforms use (e.g.
@connector_registry.decorator("name")) — importing the module is enough,
no return value is needed from it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger()


def load_project_plugins(project_dir: Path) -> list[str]:
    """Import every .py file in project_dir/plugins/ so its module-level
    registry decorators run. Returns the list of module names loaded.

    Broken plugin files are logged and skipped — one bad file must not
    prevent the rest of the project (or the engine) from starting.
    """
    plugins_dir = project_dir / "plugins"
    if not plugins_dir.is_dir():
        return []

    loaded: list[str] = []
    for py_file in sorted(plugins_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"_dex_project_plugin_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning("project plugin: could not load spec", path=str(py_file))
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded.append(module_name)
            logger.info("project plugin loaded", path=str(py_file))
        except Exception as exc:
            sys.modules.pop(module_name, None)
            logger.error("project plugin failed to load", path=str(py_file), error=str(exc))

    return loaded
