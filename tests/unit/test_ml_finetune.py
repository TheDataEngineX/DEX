"""Tests for SentenceTransformerFinetuneTrainer (dataenginex.domains.ml.training).

Kept in its own module (rather than test_ml.py) because test_ml.py has a
module-level ``pytest.importorskip("torch")`` guard for TestPyTorchTrainer,
which would skip this whole file's collection too. sentence-transformers and
torch are optional deps not installed in this dev environment, so we mock
both at the sys.modules level instead of skipping — the training call itself
is never real (per task scope: config validation / status transitions /
error handling only, no real training).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataenginex.domains.ml.training import SentenceTransformerFinetuneTrainer, train_experiment


class _FakeSentenceTransformer:
    """Stand-in for sentence_transformers.SentenceTransformer — no real model IO."""

    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path
        self.fit_kwargs: dict[str, Any] = {}

    def encode(self, sentences: Any) -> Any:
        if isinstance(sentences, str):
            return [0.1, 0.2]
        return [[0.1, 0.2] for _ in sentences]

    def fit(self, **kwargs: Any) -> None:
        self.fit_kwargs = kwargs

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class _FakeInputExample:
    def __init__(self, texts: list[str], label: float) -> None:
        self.texts = texts
        self.label = label


class _FakeLoss:
    def __init__(self, model: Any) -> None:
        self.model = model


class _FakeUtil:
    @staticmethod
    def cos_sim(a: Any, b: Any) -> list[list[float]]:
        return [[0.9]]


class _FakeDataLoader(list):  # stand-in for torch.utils.data.DataLoader
    def __init__(self, dataset: Any, shuffle: bool = False, batch_size: int = 1) -> None:
        super().__init__(dataset)


@pytest.fixture
def mock_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake sentence_transformers + torch modules (neither installed here)."""
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    st_mod.InputExample = _FakeInputExample  # type: ignore[attr-defined]
    st_mod.util = _FakeUtil()  # type: ignore[attr-defined]

    losses_mod = types.ModuleType("sentence_transformers.sentence_transformer.losses")
    losses_mod.ContrastiveLoss = _FakeLoss  # type: ignore[attr-defined]
    losses_mod.CosineSimilarityLoss = _FakeLoss  # type: ignore[attr-defined]

    st_submod = types.ModuleType("sentence_transformers.sentence_transformer")
    st_submod.losses = losses_mod  # type: ignore[attr-defined]
    st_mod.sentence_transformer = st_submod  # type: ignore[attr-defined]

    torch_mod = types.ModuleType("torch")
    torch_utils_mod = types.ModuleType("torch.utils")
    torch_utils_data_mod = types.ModuleType("torch.utils.data")
    torch_utils_data_mod.DataLoader = _FakeDataLoader  # type: ignore[attr-defined]
    torch_utils_mod.data = torch_utils_data_mod  # type: ignore[attr-defined]
    torch_mod.utils = torch_utils_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers.sentence_transformer",
        st_submod,
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers.sentence_transformer.losses",
        losses_mod,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data_mod)


class TestSentenceTransformerFinetuneTrainer:
    """Exercise the trainer with mocked sentence-transformers/torch — no real training."""

    PAIRS = [("a movie about space", "a sci-fi film"), ("a romance", "a comedy")]
    LABELS = [0.9, 0.2]

    def test_invalid_loss_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            SentenceTransformerFinetuneTrainer("m", loss_type="bogus")

    def test_predict_before_train_raises(self, mock_sentence_transformers: None) -> None:
        trainer = SentenceTransformerFinetuneTrainer("m")
        with pytest.raises(RuntimeError, match="not yet trained"):
            trainer.predict(self.PAIRS)

    def test_evaluate_before_train_raises(self, mock_sentence_transformers: None) -> None:
        trainer = SentenceTransformerFinetuneTrainer("m")
        with pytest.raises(RuntimeError, match="not yet trained"):
            trainer.evaluate(self.PAIRS, self.LABELS)

    def test_train_empty_pairs_raises(self, mock_sentence_transformers: None) -> None:
        trainer = SentenceTransformerFinetuneTrainer("m")
        with pytest.raises(ValueError, match="must not be empty"):
            trainer.train([], [])

    def test_train_mismatched_lengths_raises(self, mock_sentence_transformers: None) -> None:
        trainer = SentenceTransformerFinetuneTrainer("m")
        with pytest.raises(ValueError, match="same length"):
            trainer.train(self.PAIRS, [0.9])

    def test_train_returns_result_and_flips_status(self, mock_sentence_transformers: None) -> None:
        trainer = SentenceTransformerFinetuneTrainer(
            "review_similarity", base_model="all-MiniLM-L6-v2", loss_type="contrastive", epochs=1
        )
        assert trainer._is_fitted is False
        result = trainer.train(self.PAIRS, self.LABELS)
        assert trainer._is_fitted is True
        assert result.model_name == "review_similarity"
        assert result.parameters["base_model"] == "all-MiniLM-L6-v2"
        assert result.parameters["loss_type"] == "contrastive"
        assert "mean_cosine_similarity" in result.metrics
        assert "mean_abs_error" in result.metrics

    def test_predict_after_train(self, mock_sentence_transformers: None) -> None:
        trainer = SentenceTransformerFinetuneTrainer("m")
        trainer.train(self.PAIRS, self.LABELS)
        preds = trainer.predict(self.PAIRS)
        assert len(preds) == len(self.PAIRS)
        assert all(isinstance(p, float) for p in preds)

    def test_save_and_load(self, mock_sentence_transformers: None, tmp_path: Path) -> None:
        trainer = SentenceTransformerFinetuneTrainer("m")
        trainer.train(self.PAIRS, self.LABELS)
        out_dir = tmp_path / "finetuned-model"
        saved_path = trainer.save(str(out_dir))
        assert saved_path == str(out_dir)
        assert (out_dir / "training_metadata.json").exists()

        trainer2 = SentenceTransformerFinetuneTrainer("m")
        trainer2.load(saved_path)
        assert trainer2._is_fitted is True
        preds = trainer2.predict(self.PAIRS)
        assert len(preds) == len(self.PAIRS)

    def test_evaluate_empty_pairs(self, mock_sentence_transformers: None) -> None:
        trainer = SentenceTransformerFinetuneTrainer("m")
        trainer.train(self.PAIRS, self.LABELS)
        metrics = trainer.evaluate([], [])
        assert metrics == {"pairs_evaluated": 0.0}


def test_train_experiment_dispatches_finetune_from_lakehouse(
    mock_sentence_transformers: None,
    tmp_path: Path,
) -> None:
    silver = tmp_path / ".dex" / "lakehouse" / "silver"
    silver.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "left": ["space adventure", "romantic drama"],
                "right": ["science fiction", "love story"],
                "label": [0.9, 0.8],
            }
        ),
        silver / "review_pairs.parquet",
    )
    experiment = SimpleNamespace(
        model_type="sentence_transformer_finetune",
        params={
            "dataset_table": "review_pairs",
            "text_a_column": "left",
            "text_b_column": "right",
            "label_column": "label",
            "output_path": ".dex/models/review-similarity",
            "epochs": 1,
        },
    )
    config = SimpleNamespace(ml=SimpleNamespace(experiments={"review_similarity": experiment}))

    metrics = train_experiment(config, "review_similarity", project_dir=tmp_path)

    assert metrics["pairs_evaluated"] == 2.0
    assert (tmp_path / ".dex/models/review-similarity/training_metadata.json").exists()
