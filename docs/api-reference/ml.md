# dataenginex.ml

ML training, experiment tracking, model registry, drift detection, serving, and feature stores.

## Quick import

```python
from dataenginex.ml import (
    BaseTrainer, ModelArtifact, TrainResult,
    ModelRegistry, ModelStage,
    DriftDetector,
    ModelServer, PredictionRequest,
    SklearnTrainer,
)
```

______________________________________________________________________

## Training

`dataenginex.ml.training`

Base trainer interface and scikit-learn/XGBoost/LightGBM trainer implementations. Handles train/validation split, metric logging, and artifact serialization.

::: dataenginex.ml.training

**Key classes:** `BaseTrainer`, `SklearnTrainer`, `TrainResult`, `ModelArtifact`

```python
from dataenginex.ml.training import SklearnTrainer
from sklearn.ensemble import RandomForestClassifier

trainer = SklearnTrainer(
    experiment_name="churn",
    model=RandomForestClassifier(n_estimators=100),
)
result = trainer.train(X_train, y_train, X_val, y_val)
print(result.metrics)  # {"accuracy": 0.91, "f1": 0.88}
```

______________________________________________________________________

## Model Registry

`dataenginex.ml.registry`

Register, version, stage-transition (Staging → Production → Archived), and retrieve model artifacts.

::: dataenginex.ml.registry

**Key classes:** `ModelRegistry`, `ModelStage`, `ModelVersion`

```python
from dataenginex.ml.registry import ModelRegistry, ModelStage

registry = ModelRegistry(persist_path=".dex/models.db")
# or: ModelRegistry(store=engine.store) to share the engine's DexStore
registry.register(result.model, name="churn_v1", metrics=result.metrics)
registry.transition("churn_v1", version=1, stage=ModelStage.PRODUCTION)
model = registry.load("churn_v1", stage=ModelStage.PRODUCTION)
```

______________________________________________________________________

## MLflow Registry

`dataenginex.ml.mlflow_registry`

MLflow-backed model registry adapter. Drop-in replacement for `ModelRegistry` in MLflow-managed environments.

::: dataenginex.ml.mlflow_registry

______________________________________________________________________

## Drift Detection

`dataenginex.ml.drift`

Statistical drift detection (PSI, KS test, chi-squared) for monitoring feature and prediction distribution shift.

::: dataenginex.ml.drift

**Key class:** `DriftDetector`

```python
from dataenginex.ml.drift import DriftDetector

detector = DriftDetector(reference=X_train, method="psi")
report = detector.detect(X_new)
for feature, score in report.scores.items():
    if score > 0.2:
        print(f"Drift detected: {feature} PSI={score:.3f}")
```

______________________________________________________________________

## Serving

`dataenginex.ml.serving`

In-process model server for synchronous prediction. Loads models from the registry and wraps inference behind a typed `PredictionRequest` / `PredictionResponse` interface.

::: dataenginex.ml.serving

**Key classes:** `ModelServer`, `PredictionRequest`, `PredictionResponse`

```python
from dataenginex.ml.serving import ModelServer, PredictionRequest
from dataenginex.ml.registry import ModelStage

server = ModelServer()
server.load("churn_v1", stage=ModelStage.PRODUCTION)
response = server.predict(PredictionRequest(features={"age": 34, "tenure": 12}))
print(response.prediction, response.probability)
```

______________________________________________________________________

## Serving Engine

`dataenginex.ml.serving_engine.builtin`

Built-in serving engine wrapping `ModelServer` for `BaseServingEngine` compliance. Local-first, zero external dependencies.

::: dataenginex.ml.serving_engine.builtin

______________________________________________________________________

## Experiment Tracking

`dataenginex.ml.tracking.builtin`

Built-in JSON-backed experiment tracker. Stores experiments and runs locally — no external services required. Use MLflow via `dataenginex[tracking]` for production.

::: dataenginex.ml.tracking.builtin

```python
from dataenginex.ml.tracking.builtin import BuiltinTracker

tracker = BuiltinTracker(tracking_dir=".dex/experiments")
with tracker.start_run(experiment="churn") as run:
    run.log_param("n_estimators", 100)
    run.log_metric("accuracy", 0.91)
    run.log_artifact(model_path)
```

______________________________________________________________________

## Feature Store

`dataenginex.ml.features.builtin`

Built-in DuckDB-backed feature store. Stores feature groups as DuckDB tables.

::: dataenginex.ml.features.builtin

```python
from dataenginex.ml.features.builtin import BuiltinFeatureStore

store = BuiltinFeatureStore(database=".dex/features.duckdb")
store.write_features("user_features", df, entity_col="user_id")
features = store.get_features("user_features", entity_ids=["u1", "u2"])
```

______________________________________________________________________

## Metrics

`dataenginex.ml.metrics`

ML-specific metric computations (RMSE, MAE, AUC, F1, log-loss) independent of framework.

::: dataenginex.ml.metrics
