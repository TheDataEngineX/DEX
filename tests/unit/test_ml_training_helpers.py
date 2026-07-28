"""Tests for ML training helpers: hmac, TrainingResult, SklearnTrainer."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.dummy import DummyClassifier  # noqa: E402

from dataenginex.ml.training import (  # noqa: E402
    SklearnTrainer,
    TrainingResult,
    _hmac_sign,
    _hmac_verify,
    _SafeUnpickler,
)


class TestHmacSign:
    def test_returns_hex(self) -> None:
        sig = _hmac_sign(b"hello")
        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_deterministic(self) -> None:
        assert _hmac_sign(b"data") == _hmac_sign(b"data")

    def test_different_inputs(self) -> None:
        assert _hmac_sign(b"a") != _hmac_sign(b"b")

    def test_custom_secret(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("DEX_MODEL_SECRET", "mysecret")
        sig = _hmac_sign(b"data")
        assert len(sig) == 64


class TestHmacVerify:
    def test_valid_signature(self) -> None:
        data = b"test data"
        sig = _hmac_sign(data)
        assert _hmac_verify(data, sig) is True

    def test_invalid_signature(self) -> None:
        assert _hmac_verify(b"data", "bad") is False

    def test_tampered_data(self) -> None:
        sig = _hmac_sign(b"original")
        assert _hmac_verify(b"tampered", sig) is False


class TestSafeUnpickler:
    def test_allows_sklearn(self) -> None:
        clf = DummyClassifier()
        clf.fit([[1], [2]], [0, 1])
        data = pickle.dumps(clf)
        result = _SafeUnpickler(pickle.BytesIO(data)).load()
        assert hasattr(result, "predict")

    def test_blocks_os_module(self) -> None:
        data = pickle.dumps({"__class__": "os.system"})
        with pytest.raises(pickle.UnpicklingError, match="Unsafe pickle"):
            _SafeUnpickler(pickle.BytesIO(data)).load()


class TestTrainingResult:
    def test_defaults(self) -> None:
        r = TrainingResult(model_name="m", version="1.0.0")
        assert r.metrics == {}
        assert r.parameters == {}
        assert r.duration_seconds == 0.0
        assert r.artifact_path is None
        assert r.trained_at is not None

    def test_to_dict(self) -> None:
        r = TrainingResult(
            model_name="m",
            version="1.0.0",
            metrics={"acc": 0.95},
            parameters={"n": 10},
            duration_seconds=1.23,
            artifact_path="/tmp/model.pkl",
        )
        d = r.to_dict()
        assert d["model_name"] == "m"
        assert d["version"] == "1.0.0"
        assert d["metrics"] == {"acc": 0.95}
        assert d["duration_seconds"] == 1.23
        assert d["artifact_path"] == "/tmp/model.pkl"
        assert "trained_at" in d


class TestSklearnTrainer:
    def test_init(self) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        assert t.model_name == "mymodel"
        assert t.version == "1.0.0"
        assert t._is_fitted is False

    def test_train(self) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        X = [[1], [2], [3]]
        y = [0, 1, 0]
        result = t.train(X, y)
        assert t._is_fitted is True
        assert result.model_name == "mymodel"
        assert "train_score" in result.metrics
        assert result.duration_seconds >= 0

    def test_train_no_estimator(self) -> None:
        t = SklearnTrainer("mymodel")
        with pytest.raises(RuntimeError, match="No estimator"):
            t.train([[1]], [0])

    def test_predict(self) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        t.train([[1], [2]], [0, 1])
        preds = t.predict([[3]])
        assert len(preds) == 1

    def test_predict_not_fitted(self) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        with pytest.raises(RuntimeError, match="not yet trained"):
            t.predict([[1]])

    def test_evaluate(self) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        t.train([[1], [2], [3]], [0, 1, 0])
        metrics = t.evaluate([[4]], [1])
        assert "test_score" in metrics

    def test_evaluate_not_fitted(self) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        with pytest.raises(RuntimeError, match="not yet trained"):
            t.evaluate([[1]], [0])

    def test_save_and_load(self, tmp_path: Any) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        t.train([[1], [2]], [0, 1])
        path = str(tmp_path / "model.pkl")
        saved = t.save(path)
        assert Path(saved).exists()
        assert Path(path + ".sig").exists()
        assert Path(path + ".json").exists()

        t2 = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        t2.load(path, extra_modules=frozenset({"sklearn"}))
        assert t2._is_fitted is True
        preds = t2.predict([[3]])
        assert len(preds) == 1

    def test_save_not_fitted(self) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        with pytest.raises(RuntimeError, match="not yet trained"):
            t.save("/tmp/model.pkl")

    def test_load_tampered(self, tmp_path: Any) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        t.train([[1], [2]], [0, 1])
        path = tmp_path / "model.pkl"
        path.write_bytes(pickle.dumps(t.estimator))
        sig_path = path.with_suffix(".sig")
        sig_path.write_text("bad_signature")
        with pytest.raises(ValueError, match="HMAC verification failed"):
            t.load(str(path))

    def test_load_no_sig(self, tmp_path: Any) -> None:
        t = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        t.train([[1], [2]], [0, 1])
        path = tmp_path / "model.pkl"
        path.write_bytes(pickle.dumps(t.estimator))
        t2 = SklearnTrainer("mymodel", "1.0.0", DummyClassifier())
        t2.load(str(path))
        assert t2._is_fitted is True
