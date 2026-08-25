"""Assembly — the only layer permitted to name a concrete implementation (§5.5).

Every other layer depends on Protocols. This one turns them into objects, so
that swapping a backend is a change to one module rather than a search across
the tree. ``tests/test_architecture.py`` enforces that: no other layer may
import ``providers/``.

Profiles differ in what they wire, not in what they can do — the same published
revision runs under each (§17 Phase 6).
"""

from __future__ import annotations

from dataenginex.bootstrap.lite import build_lite_gateway, lite, open_control_store
from dataenginex.bootstrap.settings import Settings

__all__ = [
    "Settings",
    "build_lite_gateway",
    "lite",
    "open_control_store",
]
