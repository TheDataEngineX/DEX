"""DataEngineX — config-driven data, ML, and AI runtime.

The public entry point is the gateway (§13.2). Everything a client can do goes
through it, which is what lets the same code run in-process or against a remote
control plane::

    from dataenginex.bootstrap import lite

    dex = lite()
    dex.publish_revision(command, source="dex.yaml")

This module deliberately re-exports almost nothing. Before 0.6 it imported
every layer eagerly, so ``import dataenginex`` pulled in DuckDB, pyarrow, and
the model providers whether or not the caller touched them — and it made the
whole internal tree part of the public surface, which §5.5 exists to prevent.
Import the layer you need by name instead.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dataenginex")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.7.0"

__all__ = ["__version__"]
