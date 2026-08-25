"""The quality-gate failure message must name which check actually failed.

Previously it always printed completeness/uniqueness regardless of which
check(s) caused the failure — silently hiding custom_sql, schema, and
row_count_min failures, and rounding scores to 2 decimals (0.9987 displayed
as "1.00", making a real failure look like a passing score).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataenginex.config.loader import load_config
from dataenginex.domains.data.pipeline.runner import PipelineRunner
from dataenginex.foundation.errors import PipelineStepError


@pytest.fixture()
def dex_yaml(tmp_path: Path) -> Path:
    csv_file = tmp_path / "movies.csv"
    csv_file.write_text("id,title\n1,Matrix\n2,Jaws\n1,Dup\n")
    cfg = tmp_path / "dex.yaml"
    cfg.write_text(f"""\
project:
  name: test-project
data:
  sources:
    movies:
      type: csv
      connection:
        path: "{tmp_path}"
        default_file: movies.csv
  pipelines:
    ingest-movies:
      source: movies
      quality:
        uniqueness:
        - id
      target:
        layer: silver
""")
    return cfg


def test_uniqueness_failure_names_the_gate_and_shows_real_score(
    dex_yaml: Path, tmp_path: Path
) -> None:
    config = load_config(dex_yaml)
    runner = PipelineRunner(config, data_dir=tmp_path / "data")

    with pytest.raises(PipelineStepError) as exc_info:
        runner.run("ingest-movies")

    msg = str(exc_info.value)
    assert "uniqueness=0.6667" in msg
    assert "duplicates on ['id']" in msg
    # Completeness wasn't configured for this pipeline — must not appear.
    assert "completeness=" not in msg
