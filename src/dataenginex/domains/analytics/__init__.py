"""Analytics domain — transforms and quality semantics (§5.3).

A quality gate says *which* rows are acceptable; a transform says *what* the
output rows are. Both are statements about meaning, so they live here rather
than beside the engine that happens to evaluate them.
"""

from __future__ import annotations

__all__: list[str] = []
