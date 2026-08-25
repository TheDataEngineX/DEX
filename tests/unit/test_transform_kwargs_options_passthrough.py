"""Regression test: transform-specific kwargs not in TransformStepConfig's
named fields (e.g. explode's column/alias) must be nested under `options:`
in YAML, and that options dict must actually reach the transform's kwargs.
This was silently broken — options at the wrong YAML level were dropped
with no error until pipeline-run time.
"""

from __future__ import annotations

from dataenginex.config.schema import TransformStepConfig
from dataenginex.domains.data.pipeline.runner import _build_transform_kwargs


def test_options_dict_passes_through_to_transform_kwargs() -> None:
    step = TransformStepConfig(
        type="explode", options={"column": "credits.cast", "alias": "credit"}
    )

    kwargs = _build_transform_kwargs(step)

    assert kwargs == {"column": "credits.cast", "alias": "credit"}


def test_top_level_column_and_alias_are_silently_dropped_not_under_options() -> None:
    # column/alias are NOT named fields on TransformStepConfig — putting them
    # at the top level of a YAML transform step (instead of nesting under
    # options:) causes pydantic's default extra="ignore" to silently drop
    # them, and _build_transform_kwargs has no way to recover them. This
    # pins that real defect so a future config-parsing change either keeps
    # this behavior intentional and documented, or fixes it loudly.
    step = TransformStepConfig(type="explode", column="credits.cast", alias="credit")

    assert not hasattr(step, "column")
    assert not hasattr(step, "alias")
    assert _build_transform_kwargs(step) == {}
