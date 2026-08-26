"""DexEngine.trigger_ai_index_refresh — the on_pipeline_complete hook that
scheduler.py calls after any pipeline configured with
``trigger_ai_index_refresh: true`` completes. Previously this method didn't
exist at all: the call raised AttributeError, caught and logged as a warning
by the scheduler, so the lexical (Elasticsearch) index was silently never
refreshed no matter how many pipelines ran.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest

from dataenginex.engine import DexEngine


@pytest.fixture()
def dex_yaml(tmp_path: Path) -> Path:
    cfg = dedent("""\
        project:
          name: test-project
          version: "0.1.0"
        data:
          sources:
            sample_csv:
              type: csv
              path: data/sample.csv
          pipelines:
            ingest:
              source: sample_csv
              destination: raw_sample
              steps: []
        ml:
          tracking:
            backend: local
          serving:
            engine: builtin
        ai:
          agents: {}
          retrieval:
            options:
              lexical:
                backend: elasticsearch
                hosts:
                - http://elasticsearch.invalid:9200
                indices:
                  movies:
                    index_name: moviedex_movies
    """)
    p = tmp_path / "dex.yaml"
    p.write_text(cfg)
    return p


@pytest.fixture()
def engine(dex_yaml: Path) -> Generator[DexEngine]:
    eng = DexEngine(dex_yaml)
    yield eng
    eng.close()


def test_indexes_rows_from_gold_table(engine: DexEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = MagicMock()
    engine._lexical_backends["movies"] = backend
    monkeypatch.setattr(
        "dataenginex.domains.ai.tools.tool_registry.call",
        lambda *a, **k: [
            {
                "movie_id": 1,
                "title": "A",
                "overview": "o",
                "genres": "Action",
                "release_year": 2000,
                "imdb_rating": 7.5,
            },
        ],
    )

    engine.trigger_ai_index_refresh()

    backend.index.assert_called_once()
    docs = backend.index.call_args.args[0]
    assert len(docs) == 1
    assert docs[0].id == "1"
    assert "A" in docs[0].text
    assert docs[0].metadata["genres"] == "Action"


def test_noop_when_no_backend(engine: DexEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    engine._lexical_backends.pop("movies", None)
    called = MagicMock()
    monkeypatch.setattr("dataenginex.domains.ai.tools.tool_registry.call", called)

    engine.trigger_ai_index_refresh()  # must not raise

    called.assert_not_called()


def test_noop_when_query_fails(engine: DexEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = MagicMock()
    engine._lexical_backends["movies"] = backend

    def _raise(*_a: object, **_k: object) -> None:
        raise RuntimeError("table not found")

    monkeypatch.setattr("dataenginex.domains.ai.tools.tool_registry.call", _raise)

    engine.trigger_ai_index_refresh()  # must not raise

    backend.index.assert_not_called()
