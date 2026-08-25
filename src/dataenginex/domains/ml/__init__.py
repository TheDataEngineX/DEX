"""ML domain — model, training, evaluation, drift, serving semantics (§5.3).

The registry here records what a model *is* and which stage it occupies. Where
those bytes physically land is a provider concern: MLflow lives in
``providers/model/``, and this package must stay importable without it.
"""

from __future__ import annotations

__all__: list[str] = []
