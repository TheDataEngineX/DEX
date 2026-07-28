"""Tests for dataenginex.ml.serving — ModelServer and related classes."""

from __future__ import annotations

from typing import Any

from dataenginex.ml.serving import ModelServer, PredictionRequest, PredictionResponse


class _FakeModel:
    def __init__(self, predictions: list[Any] | None = None) -> None:
        self._predictions = predictions or [0.5]

    def predict(self, X: Any) -> Any:
        return self._predictions


class _FakeModelWithToList:
    """Model whose predict returns an object with .tolist() (e.g. numpy array)."""

    def __init__(self) -> None:
        pass

    def predict(self, X: Any) -> Any:
        class _Array:
            def __init__(self, data: list[Any]) -> None:
                self._data = data

            def tolist(self) -> list[Any]:
                return self._data

        return _Array([1, 2, 3])


class TestPredictionResponse:
    def test_to_dict(self) -> None:
        resp = PredictionResponse(
            model_name="m",
            version="1.0",
            predictions=[1, 2],
            latency_ms=12.345,
            request_id="r1",
        )
        d = resp.to_dict()
        assert d["model_name"] == "m"
        assert d["version"] == "1.0"
        assert d["predictions"] == [1, 2]
        assert d["latency_ms"] == 12.35
        assert d["request_id"] == "r1"
        assert "served_at" in d

    def test_to_dict_default_fields(self) -> None:
        resp = PredictionResponse(model_name="m", version="1.0")
        d = resp.to_dict()
        assert d["predictions"] == []
        assert d["latency_ms"] == 0.0
        assert d["request_id"] == ""


class TestModelServer:
    def test_load_and_predict(self) -> None:
        server = ModelServer(max_loaded=3)
        model = _FakeModel([42])
        server.load_model("clf", "1.0", model)
        req = PredictionRequest(model_name="clf", version="1.0", features=[{"x": 1}])
        resp = server.predict(req)
        assert resp.predictions == [42]
        assert resp.model_name == "clf"
        assert resp.version == "1.0"

    def test_predict_not_loaded_raises(self) -> None:
        server = ModelServer()
        req = PredictionRequest(model_name="missing", version="1.0", features=[{}])
        try:
            server.predict(req)
            raise AssertionError("should have raised")
        except RuntimeError as exc:
            assert "not loaded" in str(exc)

    def test_lru_eviction(self) -> None:
        server = ModelServer(max_loaded=2)
        server.load_model("m", "1", _FakeModel([1]))
        server.load_model("m", "2", _FakeModel([2]))
        server.load_model("m", "3", _FakeModel([3]))
        loaded = server.list_loaded()
        assert len(loaded) == 2
        assert "m:1" not in loaded

    def test_reload_same_key_moves_to_end(self) -> None:
        server = ModelServer(max_loaded=2)
        server.load_model("m", "1", _FakeModel([1]))
        server.load_model("m", "2", _FakeModel([2]))
        server.load_model("m", "1", _FakeModel([10]))
        loaded = server.list_loaded()
        assert loaded[-1] == "m:1"

    def test_list_loaded(self) -> None:
        server = ModelServer(max_loaded=5)
        server.load_model("a", "1", _FakeModel())
        server.load_model("b", "2", _FakeModel())
        assert set(server.list_loaded()) == {"a:1", "b:2"}

    def test_predict_with_toarray_model(self) -> None:
        server = ModelServer(max_loaded=3)
        server.load_model("m", "1.0", _FakeModelWithToList())
        req = PredictionRequest(model_name="m", version="1.0", features=[{"x": 1}])
        resp = server.predict(req)
        assert resp.predictions == [1, 2, 3]

    def test_predict_list_model(self) -> None:
        server = ModelServer(max_loaded=3)
        server.load_model("m", "1.0", _FakeModel([10, 20]))
        req = PredictionRequest(model_name="m", version="1.0", features=[{"x": 1}])
        resp = server.predict(req)
        assert resp.predictions == [10, 20]

    def test_resolve_production_version_from_loaded(self) -> None:
        server = ModelServer(max_loaded=3)
        server.load_model("m", "2.0", _FakeModel())
        version = server._resolve_production_version("m")
        assert version == "2.0"

    def test_resolve_production_version_no_model_raises(self) -> None:
        server = ModelServer()
        try:
            server._resolve_production_version("ghost")
            raise AssertionError("should have raised")
        except RuntimeError as exc:
            assert "No version found" in str(exc)

    def test_resolve_production_version_from_registry(self) -> None:
        class _FakeRegistry:
            def get_production(self, name: str) -> Any:
                class _Prod:
                    version = "3.0"

                return _Prod()

        server = ModelServer(registry=_FakeRegistry(), max_loaded=3)
        version = server._resolve_production_version("m")
        assert version == "3.0"

    def test_empty_features(self) -> None:
        server = ModelServer(max_loaded=3)
        server.load_model("m", "1.0", _FakeModel([0]))
        req = PredictionRequest(model_name="m", version="1.0", features=[])
        resp = server.predict(req)
        assert resp.predictions == [0]

    def test_predict_uses_version_param(self) -> None:
        server = ModelServer(max_loaded=3)
        server.load_model("m", "1.0", _FakeModel([100]))
        req = PredictionRequest(model_name="m", version="1.0", features=[{"x": 1}])
        resp = server.predict(req)
        assert resp.predictions == [100]


class TestFeaturesToArray:
    def test_empty(self) -> None:
        assert ModelServer._features_to_array([]) == []

    def test_single_row(self) -> None:
        result = ModelServer._features_to_array([{"a": 1, "b": 2}])
        assert result == [[1, 2]]

    def test_multiple_rows(self) -> None:
        result = ModelServer._features_to_array([{"x": 1}, {"x": 2}])
        assert result == [[1], [2]]

    def test_missing_key_fills_none(self) -> None:
        result = ModelServer._features_to_array([{"a": 1, "b": 10}, {"a": 2}])
        assert result[0] == [1, 10]
        assert result[1] == [2, None]
