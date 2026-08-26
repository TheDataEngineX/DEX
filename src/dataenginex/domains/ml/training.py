"""
Training framework — abstract trainer and scikit-learn implementation.

The ``BaseTrainer`` ABC defines a standard train → evaluate → save lifecycle.
``SklearnTrainer`` provides a concrete implementation using scikit-learn
(when installed; otherwise a stub that raises ``ImportError``).
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import structlog

from dataenginex import _json

logger = structlog.get_logger()
__all__ = [
    "BaseTrainer",
    "PyTorchTrainer",
    "SentenceTransformerFinetuneTrainer",
    "SklearnTrainer",
    "TrainingResult",
]

# ---------------------------------------------------------------------------
# Pickle safety — restrict deserialization to trusted namespaces
# ---------------------------------------------------------------------------

_ALLOWED_PICKLE_MODULES: frozenset[str] = frozenset(
    {
        "sklearn",
        "numpy",
        "scipy",
        "copy",
        "builtins",
        "collections",
        "_codecs",
    }
)


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that only permits classes from trusted ML namespaces."""

    def __init__(
        self,
        fp: io.BytesIO,
        *,
        extra_modules: frozenset[str] | None = None,
    ) -> None:
        super().__init__(fp)
        self._allowed = _ALLOWED_PICKLE_MODULES | (extra_modules or frozenset())

    def find_class(
        self,
        module: str,
        name: str,
    ) -> type:
        top_level = module.split(".")[0]
        if top_level not in self._allowed:
            msg = (
                f"Unsafe pickle: module '{module}' is not in the "
                f"allowed list {sorted(self._allowed)}"
            )
            raise pickle.UnpicklingError(msg)
        return super().find_class(module, name)  # type: ignore[no-any-return]


def _hmac_sign(data: bytes) -> str:
    """Compute HMAC-SHA256 of *data* using DEX_MODEL_SECRET or a default key."""
    secret = os.environ.get("DEX_MODEL_SECRET", "dex-dev-only").encode()
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


def _hmac_verify(data: bytes, expected_sig: str) -> bool:
    """Return ``True`` if HMAC signature matches."""
    actual = _hmac_sign(data)
    return hmac.compare_digest(actual, expected_sig)


@dataclass
class TrainingResult:
    """Outcome of a model training run.

    Attributes:
        model_name: Name of the trained model.
        version: Semantic version of this training run.
        metrics: Training metrics (e.g. ``{"train_score": 0.95}``).
        parameters: Hyper-parameters used for training.
        duration_seconds: Wall-clock training time.
        artifact_path: Path where the model artifact is saved.
        trained_at: Timestamp of training completion.
    """

    model_name: str
    version: str
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    artifact_path: str | None = None
    trained_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the training result to a plain dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "duration_seconds": round(self.duration_seconds, 2),
            "artifact_path": self.artifact_path,
            "trained_at": self.trained_at.isoformat(),
        }


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""

    def __init__(self, model_name: str, version: str = "1.0.0") -> None:
        self.model_name = model_name
        self.version = version

    @abstractmethod
    def train(
        self,
        X_train: Any,
        y_train: Any,
        **params: Any,  # noqa: N803
    ) -> TrainingResult:
        """Train the model and return metrics."""
        ...

    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any) -> dict[str, float]:  # noqa: N803
        """Evaluate the model on test data."""
        ...

    @abstractmethod
    def predict(self, X: Any) -> Any:  # noqa: N803
        """Generate predictions."""
        ...

    @abstractmethod
    def save(self, path: str) -> str:
        """Persist the model to *path* and return the artifact path."""
        ...

    @abstractmethod
    def load(
        self,
        path: str,
        *,
        extra_modules: frozenset[str] | None = None,
    ) -> None:
        """Load a previously saved model from *path*."""
        ...


class SklearnTrainer(BaseTrainer):
    """scikit-learn model trainer.

    Works with any sklearn estimator (or pipeline) that implements
    ``fit``, ``predict``, and ``score``.

    Parameters
    ----------
    model_name:
        Name used in model registry.
    version:
        Semantic version string.
    estimator:
        An sklearn estimator instance (e.g. ``RandomForestClassifier()``).
    """

    def __init__(
        self,
        model_name: str,
        version: str = "1.0.0",
        estimator: Any = None,
    ) -> None:
        super().__init__(model_name, version)
        self.estimator = estimator
        self._is_fitted = False

    def train(
        self,
        X_train: Any,
        y_train: Any,
        **params: Any,  # noqa: N803
    ) -> TrainingResult:
        """Fit the estimator on *X_train*/*y_train* and return metrics."""
        if self.estimator is None:
            raise RuntimeError("No estimator provided to SklearnTrainer")

        # Apply params
        if params:
            self.estimator.set_params(**params)

        start = time.perf_counter()
        self.estimator.fit(X_train, y_train)
        duration = time.perf_counter() - start
        self._is_fitted = True

        # Compute training score
        train_score = float(self.estimator.score(X_train, y_train))
        metrics = {"train_score": round(train_score, 4)}

        logger.info(
            "Trained %s v%s in %.2fs (train_score=%.4f)",
            self.model_name,
            self.version,
            duration,
            train_score,
        )

        return TrainingResult(
            model_name=self.model_name,
            version=self.version,
            metrics=metrics,
            parameters=self.estimator.get_params(),
            duration_seconds=duration,
        )

    def evaluate(self, X_test: Any, y_test: Any) -> dict[str, float]:  # noqa: N803
        """Score the fitted model on *X_test*/*y_test* and return metrics."""
        if not self._is_fitted:
            raise RuntimeError("Model not yet trained")

        test_score = float(self.estimator.score(X_test, y_test))
        predictions = self.estimator.predict(X_test)

        metrics: dict[str, float] = {"test_score": round(test_score, 4)}

        # Attempt classification metrics
        try:
            from sklearn.metrics import (
                f1_score,
                precision_score,
                recall_score,
            )

            metrics["precision"] = round(
                float(
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                ),
                4,
            )
            metrics["recall"] = round(
                float(
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                ),
                4,
            )
            metrics["f1"] = round(
                float(
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                ),
                4,
            )
        except ImportError:
            logger.debug(
                "sklearn.metrics not available — skipping precision/recall/f1",
            )

        return metrics

    def predict(self, X: Any) -> Any:  # noqa: N803
        """Generate predictions for *X* using the fitted estimator."""
        if not self._is_fitted:
            raise RuntimeError("Model not yet trained")
        return self.estimator.predict(X)

    def save(self, path: str) -> str:
        """Pickle the fitted model, write an HMAC signature, and metadata."""
        if not self._is_fitted:
            raise RuntimeError("Model not yet trained")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        model_bytes = pickle.dumps(self.estimator)
        out.write_bytes(model_bytes)

        # HMAC sidecar — verifies integrity on load. The suffix is appended
        # rather than substituted: ``with_suffix`` would turn both "model.pkl"
        # and "model.onnx" into "model.sig", so two artifacts in one directory
        # would silently overwrite each other's signature.
        sig_path = out.with_name(out.name + ".sig")
        sig_path.write_text(_hmac_sign(model_bytes))

        # Save metadata alongside
        meta = out.with_name(out.name + ".json")
        meta.write_text(
            _json.dumps(
                {
                    "model_name": self.model_name,
                    "version": self.version,
                    "saved_at": datetime.now(tz=UTC).isoformat(),
                }
            )
        )

        logger.info("model saved", name=self.model_name, path=str(out))
        return str(out)

    def load(
        self,
        path: str,
        *,
        extra_modules: frozenset[str] | None = None,
    ) -> None:
        """Load a pickled model with HMAC verification and safe unpickling.

        Args:
            path: Filesystem path to the ``.pkl`` artifact.
            extra_modules: Additional top-level module names to allow
                during unpickling (e.g. ``frozenset({"tests"})`` for
                test-only estimators).
        """
        artifact = Path(path)
        data = artifact.read_bytes()

        # Verify HMAC signature if sidecar exists
        sig_path = artifact.with_name(artifact.name + ".sig")
        if sig_path.exists():
            expected = sig_path.read_text().strip()
            if not _hmac_verify(data, expected):
                msg = (
                    f"HMAC verification failed for {path}. "
                    "The model file may have been tampered with."
                )
                raise ValueError(msg)
        else:
            logger.warning(
                "No .sig sidecar for %s — skipping HMAC check",
                path,
            )

        # Safe unpickle — restricted to sklearn/numpy namespaces
        self.estimator = _SafeUnpickler(
            io.BytesIO(data),
            extra_modules=extra_modules,
        ).load()
        self._is_fitted = True
        logger.info("model loaded", name=self.model_name, path=str(path))


class PyTorchTrainer(BaseTrainer):
    """PyTorch model trainer.

    Works with any ``torch.nn.Module`` that accepts a single float tensor input
    and returns a single float tensor output.  All ``torch`` imports are
    deferred so the class can be imported without PyTorch installed.

    Parameters
    ----------
    model_name:
        Name used in model registry.
    version:
        Semantic version string.
    model:
        A ``torch.nn.Module`` instance.
    optimizer_cls:
        Optimizer class (default: ``torch.optim.Adam``).
    criterion:
        Loss function instance (default: ``torch.nn.MSELoss()``).
    lr:
        Learning rate (default: ``1e-3``).
    epochs:
        Training epochs (default: ``10``).
    batch_size:
        Mini-batch size (default: ``32``).
    device:
        Torch device string — ``"cpu"`` or ``"cuda"`` (default: ``"cpu"``).
    """

    def __init__(
        self,
        model_name: str,
        version: str = "1.0.0",
        model: Any = None,
        optimizer_cls: Any = None,
        criterion: Any = None,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        super().__init__(model_name, version)
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self._is_fitted = False
        self._optimizer: Any = None

    def _to_tensor(self, data: Any, dtype: Any) -> Any:
        import torch

        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype, device=self.device)
        return torch.tensor(data, dtype=dtype, device=self.device)

    def train(
        self,
        X_train: Any,
        y_train: Any,
        **params: Any,  # noqa: N803
    ) -> TrainingResult:
        """Train *model* on *X_train*/*y_train* and return metrics."""
        import torch

        if self.model is None:
            raise RuntimeError("No model provided to PyTorchTrainer")

        epochs = int(params.get("epochs", self.epochs))
        batch_size = int(params.get("batch_size", self.batch_size))
        lr = float(params.get("lr", self.lr))

        criterion = self.criterion if self.criterion is not None else torch.nn.MSELoss()
        optimizer_cls = self.optimizer_cls or torch.optim.Adam
        self._optimizer = optimizer_cls(self.model.parameters(), lr=lr)

        self.model.to(self.device)
        self.model.train()

        X = self._to_tensor(X_train, torch.float32)
        y = self._to_tensor(y_train, torch.float32)
        n = X.shape[0]

        start = time.perf_counter()
        last_loss = float("inf")
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            steps = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X[idx], y[idx]
                self._optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out.squeeze(-1), yb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
            last_loss = epoch_loss / max(steps, 1)
        duration = time.perf_counter() - start

        self._is_fitted = True
        metrics = {"train_loss": round(last_loss, 6)}
        logger.info(
            "PyTorchTrainer: trained %s v%s in %.2fs (train_loss=%.6f)",
            self.model_name,
            self.version,
            duration,
            last_loss,
        )
        return TrainingResult(
            model_name=self.model_name,
            version=self.version,
            metrics=metrics,
            parameters={"epochs": epochs, "batch_size": batch_size, "lr": lr},
            duration_seconds=duration,
        )

    def evaluate(self, X_test: Any, y_test: Any) -> dict[str, float]:  # noqa: N803
        """Compute test loss (and accuracy when output is multi-class) on *X_test*/*y_test*."""
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model not yet trained")

        criterion = self.criterion if self.criterion is not None else torch.nn.MSELoss()
        X = self._to_tensor(X_test, torch.float32)
        y = self._to_tensor(y_test, torch.float32)

        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
            loss = criterion(out.squeeze(-1), y)

        metrics: dict[str, float] = {"test_loss": round(float(loss.item()), 6)}

        # Accuracy — only meaningful for multi-class output (ndim > 1, classes > 1)
        if out.ndim > 1 and out.shape[-1] > 1:
            preds = out.argmax(dim=-1)
            y_long = self._to_tensor(y_test, torch.long)
            acc = float((preds == y_long).float().mean().item())
            metrics["accuracy"] = round(acc, 4)

        return metrics

    def predict(self, X: Any) -> Any:  # noqa: N803
        """Return model output for *X* as a numpy array."""
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model not yet trained")

        self.model.eval()
        with torch.no_grad():
            out = self.model(self._to_tensor(X, torch.float32))
        return out.cpu().detach().numpy()

    def save(self, path: str) -> str:
        """Save ``model.state_dict()`` with an HMAC sidecar and metadata JSON."""
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model not yet trained")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        model_bytes = buf.getvalue()
        out.write_bytes(model_bytes)

        out.with_suffix(".sig").write_text(_hmac_sign(model_bytes))
        out.with_suffix(".json").write_text(
            _json.dumps(
                {
                    "model_name": self.model_name,
                    "version": self.version,
                    "saved_at": datetime.now(tz=UTC).isoformat(),
                }
            )
        )

        logger.info("pytorch model saved", name=self.model_name, path=str(out))
        return str(out)

    def load(
        self,
        path: str,
        *,
        extra_modules: frozenset[str] | None = None,
    ) -> None:
        """Load ``state_dict`` from *path* with HMAC verification.

        The model architecture (``self.model``) must already be set; only
        weights are restored.  *extra_modules* is accepted for interface
        compatibility but unused — PyTorch uses ``weights_only=True`` instead.
        """
        import torch

        del extra_modules  # unused — safety handled by weights_only=True

        if self.model is None:
            raise RuntimeError("model architecture must be set on PyTorchTrainer before load()")

        artifact = Path(path)
        data = artifact.read_bytes()

        sig_path = artifact.with_suffix(".sig")
        if sig_path.exists():
            expected = sig_path.read_text().strip()
            if not _hmac_verify(data, expected):
                msg = (
                    f"HMAC verification failed for {path}. "
                    "The model file may have been tampered with."
                )
                raise ValueError(msg)
        else:
            logger.warning("No .sig sidecar for %s — skipping HMAC check", path)

        state = torch.load(io.BytesIO(data), weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._is_fitted = True
        logger.info("pytorch model loaded", name=self.model_name, path=str(path))


_VALID_LOSS_TYPES = frozenset({"contrastive", "cosine"})


class SentenceTransformerFinetuneTrainer(BaseTrainer):
    """Fine-tunes a sentence-transformers bi-encoder on labeled sentence pairs.

    Generic similarity-finetuning job type — domain-agnostic. Training data is
    a set of ``(sentence_a, sentence_b)`` pairs plus a parallel list of
    similarity labels (typically floats in ``[0, 1]``, or 0/1 for a binary
    match/no-match signal). Runs via sentence-transformers' training API
    (``SentenceTransformer.fit`` + a pairwise loss).

    Args:
        model_name: Registry name for the resulting fine-tuned model.
        version: Semantic version.
        base_model: Pretrained sentence-transformers model name or local path
            to start fine-tuning from (e.g. ``"all-MiniLM-L6-v2"``).
        loss_type: ``"contrastive"`` (default) or ``"cosine"``.
        epochs: Fine-tuning epochs.
        batch_size: Mini-batch size.
        warmup_steps: LR warmup steps.
    """

    def __init__(
        self,
        model_name: str,
        version: str = "1.0.0",
        *,
        base_model: str = "all-MiniLM-L6-v2",
        loss_type: str = "contrastive",
        epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__(model_name, version)
        if loss_type not in _VALID_LOSS_TYPES:
            msg = f"Unsupported loss_type {loss_type!r}. Must be one of {sorted(_VALID_LOSS_TYPES)}"
            raise ValueError(msg)
        self.base_model = base_model
        self.loss_type = loss_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self._model: Any = None
        self._is_fitted = False

    def _build_loss(self, model: Any) -> Any:
        from sentence_transformers.sentence_transformer import losses

        if self.loss_type == "cosine":
            return losses.CosineSimilarityLoss(model)
        return losses.ContrastiveLoss(model)

    def train(
        self,
        X_train: Any,  # noqa: N803
        y_train: Any,
        **params: Any,
    ) -> TrainingResult:
        """Fine-tune on sentence pairs.

        Args:
            X_train: Sequence of ``(sentence_a, sentence_b)`` pairs.
            y_train: Parallel sequence of similarity labels.
        """
        from sentence_transformers import InputExample, SentenceTransformer
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset as TorchDataset

        if not X_train:
            raise ValueError("X_train (training pairs) must not be empty")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")

        epochs = int(params.get("epochs", self.epochs))
        batch_size = int(params.get("batch_size", self.batch_size))
        warmup_steps = int(params.get("warmup_steps", self.warmup_steps))

        self._model = SentenceTransformer(self.base_model)
        examples: list[InputExample] = [
            InputExample(texts=[a, b], label=float(label))
            for (a, b), label in zip(X_train, y_train, strict=True)
        ]

        class _ListDataset(TorchDataset[InputExample]):
            def __init__(self, items: list[InputExample]) -> None:
                self._items = items
            def __getitem__(self, idx: int) -> InputExample:
                return self._items[idx]
            def __len__(self) -> int:
                return len(self._items)

        loader: DataLoader[InputExample] = DataLoader(
            _ListDataset(examples), shuffle=True, batch_size=batch_size,
        )
        loss_fn = self._build_loss(self._model)

        start = time.perf_counter()
        self._model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=False,
        )
        duration = time.perf_counter() - start
        self._is_fitted = True

        metrics = self.evaluate(X_train, y_train)
        logger.info(
            "Finetuned sentence-transformer %s v%s in %.2fs (base=%s, loss=%s)",
            self.model_name,
            self.version,
            duration,
            self.base_model,
            self.loss_type,
        )
        return TrainingResult(
            model_name=self.model_name,
            version=self.version,
            metrics=metrics,
            parameters={
                "base_model": self.base_model,
                "loss_type": self.loss_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "warmup_steps": warmup_steps,
            },
            duration_seconds=duration,
        )

    def evaluate(self, X_test: Any, y_test: Any) -> dict[str, float]:  # noqa: N803
        """Compute mean cosine similarity + mean absolute error vs *y_test* labels."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not yet trained")
        if not X_test:
            return {"pairs_evaluated": 0.0}

        from sentence_transformers import util

        sims: list[float] = []
        for a, b in X_test:
            emb = self._model.encode([a, b])
            sims.append(float(util.cos_sim(emb[0], emb[1])[0][0]))

        mae = sum(abs(s - float(y)) for s, y in zip(sims, y_test, strict=True)) / len(sims)
        return {
            "pairs_evaluated": float(len(sims)),
            "mean_cosine_similarity": round(sum(sims) / len(sims), 4),
            "mean_abs_error": round(mae, 4),
        }

    def predict(self, X: Any) -> Any:  # noqa: N803
        """Return cosine similarity scores for a sequence of sentence pairs."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not yet trained")

        from sentence_transformers import util

        return [
            float(util.cos_sim(self._model.encode(a), self._model.encode(b))[0][0]) for a, b in X
        ]

    def save(self, path: str) -> str:
        """Save the fine-tuned model directory + metadata sidecar.

        Unlike the pickle-based trainers above, sentence-transformers persists
        a directory of framework-native config/weight files rather than a
        single pickle blob, so the HMAC-over-pickle scheme doesn't apply here.
        # ponytail: no tamper-detection sidecar for this artifact type; add
        # directory checksums if shared-artifact integrity becomes a concern.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not yet trained")

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save(str(out))

        meta = out / "training_metadata.json"
        meta.write_text(
            _json.dumps(
                {
                    "model_name": self.model_name,
                    "version": self.version,
                    "base_model": self.base_model,
                    "loss_type": self.loss_type,
                    "saved_at": datetime.now(tz=UTC).isoformat(),
                }
            )
        )
        logger.info("sentence-transformer model saved", name=self.model_name, path=str(out))
        return str(out)

    def load(
        self,
        path: str,
        *,
        extra_modules: frozenset[str] | None = None,
    ) -> None:
        """Load a previously fine-tuned model directory."""
        from sentence_transformers import SentenceTransformer

        _ = extra_modules  # SentenceTransformer handles its own module graph
        self._model = SentenceTransformer(path)
        self._is_fitted = True
        logger.info("sentence-transformer model loaded", name=self.model_name, path=path)


def _load_lakehouse_rows(project_dir: Path, table_name: str) -> list[dict[str, Any]]:
    for layer in ("silver", "gold", "bronze"):
        layer_dir = project_dir / ".dex" / "lakehouse" / layer
        parquet_path = layer_dir / f"{table_name}.parquet"
        if parquet_path.exists():
            import pyarrow.parquet as pq

            table = pq.read_table(parquet_path)
            return cast(list[dict[str, Any]], table.to_pylist())
        delta_path = layer_dir / table_name
        if (delta_path / "_delta_log").exists():
            from dataenginex.providers.object_store.storage import DeltaStorage

            return DeltaStorage(base_path=str(layer_dir)).read(table_name) or []
    raise FileNotFoundError(f"Lakehouse table not found: {table_name}")


def train_experiment(
    config: Any,
    experiment_name: str,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a named experiment from dex.yaml and return metrics dict."""
    exp_cfg = config.ml.experiments[experiment_name]
    if exp_cfg.model_type == "sentence_transformer_finetune":
        params = dict(exp_cfg.params)
        table_name = str(params.pop("dataset_table"))
        text_a = str(params.pop("text_a_column"))
        text_b = str(params.pop("text_b_column"))
        label = str(params.pop("label_column"))
        output_path = Path(str(params.pop("output_path", f".dex/models/{experiment_name}")))
        root = Path(project_dir or Path.cwd()).resolve()
        rows = _load_lakehouse_rows(root, table_name)
        pairs = [
            (str(row[text_a]), str(row[text_b]))
            for row in rows
            if row.get(text_a) is not None
            and row.get(text_b) is not None
            and row.get(label) is not None
        ]
        labels = [
            float(row[label])
            for row in rows
            if row.get(text_a) is not None
            and row.get(text_b) is not None
            and row.get(label) is not None
        ]
        sentence_trainer = SentenceTransformerFinetuneTrainer(
            experiment_name,
            base_model=str(params.pop("base_model", "all-MiniLM-L6-v2")),
            loss_type=str(params.pop("loss_type", "contrastive")),
            epochs=int(params.pop("epochs", 3)),
            batch_size=int(params.pop("batch_size", 16)),
            warmup_steps=int(params.pop("warmup_steps", 100)),
        )
        result = sentence_trainer.train(pairs, labels)
        sentence_trainer.save(str(output_path if output_path.is_absolute() else root / output_path))
        return result.metrics
    sklearn_trainer = SklearnTrainer(model_name=experiment_name)
    result = sklearn_trainer.train([], [], algorithm=getattr(exp_cfg, "algorithm", "auto"))
    return result.metrics
