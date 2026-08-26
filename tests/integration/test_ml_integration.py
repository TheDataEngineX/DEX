"""Integration tests — ML tracking, feature store, and cross-subsystem flows."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataenginex.domains.ml.features.builtin import BuiltinFeatureStore
from dataenginex.domains.ml.tracking.builtin import BuiltinTracker

# ---------------------------------------------------------------------------
# BuiltinTracker — full experiment lifecycle
# ---------------------------------------------------------------------------


class TestBuiltinTrackerLifecycle:
    def test_create_experiment_returns_id(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        exp_id = tracker.create_experiment("churn_v1")
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0

    def test_duplicate_experiment_name_is_idempotent(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        id1 = tracker.create_experiment("same_name")
        id2 = tracker.create_experiment("same_name")
        assert id1 == id2

    def test_list_experiments(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        tracker.create_experiment("exp_a")
        tracker.create_experiment("exp_b")
        experiments = tracker.list_experiments()
        names = [e["name"] for e in experiments]
        assert "exp_a" in names
        assert "exp_b" in names

    def test_start_run_returns_id(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        exp_id = tracker.create_experiment("test_exp")
        run_id = tracker.start_run(exp_id, run_name="run-1")
        assert isinstance(run_id, str)

    def test_start_run_status_is_running(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        exp_id = tracker.create_experiment("test_exp")
        run_id = tracker.start_run(exp_id)
        runs = tracker.list_runs(exp_id)
        run = next(r for r in runs if r["run_id"] == run_id)
        assert run["status"] == "RUNNING"

    def test_log_params_and_retrieve(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        exp_id = tracker.create_experiment("test_exp")
        run_id = tracker.start_run(exp_id)
        tracker.log_params(run_id, {"learning_rate": 0.001, "epochs": 50})
        runs = tracker.list_runs(exp_id)
        run = next(r for r in runs if r["run_id"] == run_id)
        assert run["params"]["learning_rate"] == 0.001
        assert run["params"]["epochs"] == 50

    def test_log_metrics_multiple_steps(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        exp_id = tracker.create_experiment("metrics_exp")
        run_id = tracker.start_run(exp_id)
        tracker.log_metrics(run_id, {"accuracy": 0.80}, step=1)
        tracker.log_metrics(run_id, {"accuracy": 0.85}, step=2)
        tracker.log_metrics(run_id, {"accuracy": 0.90}, step=3)
        runs = tracker.list_runs(exp_id)
        run = next(r for r in runs if r["run_id"] == run_id)
        assert len(run["metrics"]["accuracy"]) == 3
        values = [entry["value"] for entry in run["metrics"]["accuracy"]]
        assert values == [0.80, 0.85, 0.90]

    def test_end_run_updates_status(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        exp_id = tracker.create_experiment("test_exp")
        run_id = tracker.start_run(exp_id)
        tracker.end_run(run_id, status="FINISHED")
        runs = tracker.list_runs(exp_id)
        run = next(r for r in runs if r["run_id"] == run_id)
        assert run["status"] == "FINISHED"
        assert run["ended_at"] is not None

    def test_failed_run_status(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        exp_id = tracker.create_experiment("fail_exp")
        run_id = tracker.start_run(exp_id)
        tracker.end_run(run_id, status="FAILED")
        runs = tracker.list_runs(exp_id)
        run = next(r for r in runs if r["run_id"] == run_id)
        assert run["status"] == "FAILED"

    def test_start_run_unknown_experiment_raises(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        with pytest.raises(KeyError, match="not found"):
            tracker.start_run("nonexistent_exp_id")

    def test_log_params_unknown_run_raises(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        with pytest.raises(KeyError, match="not found"):
            tracker.log_params("bad_run_id", {"x": 1})

    def test_end_run_unknown_run_raises(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        with pytest.raises(KeyError, match="not found"):
            tracker.end_run("bad_run_id")

    def test_multiple_runs_per_experiment(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        exp_id = tracker.create_experiment("multi_run")
        run_ids = [tracker.start_run(exp_id, run_name=f"run-{i}") for i in range(3)]
        runs = tracker.list_runs(exp_id)
        assert len(runs) == 3
        found_ids = {r["run_id"] for r in runs}
        assert set(run_ids) == found_ids


class TestBuiltinTrackerPersistence:
    def test_experiments_survive_restart(self, tmp_path: Path) -> None:
        storage = str(tmp_path / "tracking")
        tracker1 = BuiltinTracker(storage_dir=storage)
        exp_id = tracker1.create_experiment("persistent_exp")
        tracker1.start_run(exp_id, run_name="run-0")

        # Reload from same directory
        tracker2 = BuiltinTracker(storage_dir=storage)
        experiments = tracker2.list_experiments()
        assert any(e["name"] == "persistent_exp" for e in experiments)

    def test_runs_survive_restart(self, tmp_path: Path) -> None:
        storage = str(tmp_path / "tracking")
        tracker1 = BuiltinTracker(storage_dir=storage)
        exp_id = tracker1.create_experiment("exp")
        run_id = tracker1.start_run(exp_id)
        tracker1.log_metrics(run_id, {"loss": 0.5})

        tracker2 = BuiltinTracker(storage_dir=storage)
        runs = tracker2.list_runs(exp_id)
        assert any(r["run_id"] == run_id for r in runs)


# ---------------------------------------------------------------------------
# BuiltinFeatureStore
# ---------------------------------------------------------------------------


class TestBuiltinFeatureStore:
    def test_save_and_retrieve_features(self, tmp_path: Path) -> None:
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        data = [
            {"user_id": "u1", "age": 25, "spend": 100.0},
            {"user_id": "u2", "age": 30, "spend": 200.0},
            {"user_id": "u3", "age": 35, "spend": 300.0},
        ]
        fs.save_features("user_features", data, entity_key="user_id")
        retrieved = fs.get_features("user_features", ["u1", "u3"])
        assert len(retrieved) == 2
        ids = {r["user_id"] for r in retrieved}
        assert ids == {"u1", "u3"}
        fs.close()

    def test_list_feature_groups(self, tmp_path: Path) -> None:
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        fs.save_features("group_a", [{"id": "1", "val": 1}], entity_key="id")
        fs.save_features("group_b", [{"id": "2", "val": 2}], entity_key="id")
        groups = fs.list_feature_groups()
        assert "group_a" in groups
        assert "group_b" in groups
        fs.close()

    def test_get_features_unknown_group_raises(self, tmp_path: Path) -> None:
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        with pytest.raises(KeyError, match="not found"):
            fs.get_features("nonexistent_group", ["e1"])
        fs.close()

    def test_save_features_overwrites_existing(self, tmp_path: Path) -> None:
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        fs.save_features("grp", [{"id": "1", "val": 10}], entity_key="id")
        fs.save_features("grp", [{"id": "1", "val": 99}], entity_key="id")
        result = fs.get_features("grp", ["1"])
        assert result[0]["val"] == 99
        fs.close()

    def test_save_empty_list_is_noop(self, tmp_path: Path) -> None:
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        # Should not raise
        fs.save_features("empty_grp", [], entity_key="id")
        fs.close()

    def test_get_features_returns_empty_for_missing_entities(self, tmp_path: Path) -> None:
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        fs.save_features("grp", [{"id": "1", "val": 1}], entity_key="id")
        result = fs.get_features("grp", ["999"])
        assert result == []
        fs.close()

    def test_save_pyarrow_table(self, tmp_path: Path) -> None:
        import pyarrow as pa

        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        tbl = pa.table({"id": ["a", "b"], "score": [0.9, 0.7]})
        fs.save_features("arrow_group", tbl, entity_key="id")
        result = fs.get_features("arrow_group", ["a"])
        assert len(result) == 1
        assert result[0]["id"] == "a"
        fs.close()

    def test_unsupported_data_type_raises(self, tmp_path: Path) -> None:
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        with pytest.raises(TypeError, match="Unsupported data type"):
            fs.save_features("bad", {"not": "a list"}, entity_key="id")  # type: ignore[arg-type]
        fs.close()


# ---------------------------------------------------------------------------
# Cross-subsystem: Tracker + FeatureStore together
# ---------------------------------------------------------------------------


class TestTrackerAndFeatureStoreCrossSubsystem:
    def test_experiment_run_with_feature_persistence(self, tmp_path: Path) -> None:
        """An ML run logs metrics AND saves the feature group used — both should persist."""
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))

        # 1. Create experiment + start run
        exp_id = tracker.create_experiment("churn_model")
        run_id = tracker.start_run(exp_id, run_name="baseline")

        # 2. Log hyperparams
        tracker.log_params(run_id, {"n_estimators": 100, "max_depth": 5})

        # 3. Save features used in this run
        feature_data = [
            {"customer_id": "c1", "tenure": 12, "monthly_charges": 65.0},
            {"customer_id": "c2", "tenure": 5, "monthly_charges": 90.0},
        ]
        fs.save_features("churn_features_v1", feature_data, entity_key="customer_id")

        # 4. Log metrics and end run
        tracker.log_metrics(run_id, {"accuracy": 0.84, "f1": 0.81})
        tracker.end_run(run_id, status="FINISHED")

        # 5. Verify tracker state
        runs = tracker.list_runs(exp_id)
        run = next(r for r in runs if r["run_id"] == run_id)
        assert run["status"] == "FINISHED"
        assert run["params"]["n_estimators"] == 100
        assert run["metrics"]["accuracy"][0]["value"] == 0.84

        # 6. Verify feature store state
        features = fs.get_features("churn_features_v1", ["c1"])
        assert features[0]["tenure"] == 12

        fs.close()

    def test_multiple_experiments_with_separate_feature_groups(self, tmp_path: Path) -> None:
        tracker = BuiltinTracker(storage_dir=str(tmp_path / "tracking"))
        fs = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))

        for i in range(3):
            exp_id = tracker.create_experiment(f"model_v{i}")
            run_id = tracker.start_run(exp_id)
            tracker.log_metrics(run_id, {"accuracy": 0.7 + i * 0.05})
            tracker.end_run(run_id)
            fs.save_features(
                f"features_v{i}",
                [{"id": f"e{i}", "value": i}],
                entity_key="id",
            )

        assert len(tracker.list_experiments()) == 3
        assert len(fs.list_feature_groups()) == 3
        fs.close()
