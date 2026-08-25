# Training

Model training with sklearn, PyTorch, and sentence transformers.

```python
from dataenginex.domains.ml.training import SklearnTrainer

trainer = SklearnTrainer(model_name="price_predictor", version="1.0.0")
result = trainer.train(X_train, y_train, algorithm="random_forest")
```

## BaseTrainer (ABC)

```python
BaseTrainer(model_name: str, version: str = "1.0.0")
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `train` | `(X_train, y_train, **params) -> TrainingResult` | Train the model (abstract) |
| `evaluate` | `(X_test, y_test) -> dict[str, float]` | Evaluate metrics (abstract) |
| `predict` | `(X) -> Any` | Run inference (abstract) |
| `save` | `(path: str) -> str` | Save model to disk (abstract) |
| `load` | `(path, extra_modules=None) -> None` | Load model from disk (abstract) |

## SklearnTrainer

```python
SklearnTrainer(model_name: str, version: str = "1.0.0", estimator: Any = None)
```

Trains scikit-learn estimators. Supports: random_forest, gradient_boosting, logistic_regression, linear_regression, xgboost (if installed).

## PyTorchTrainer

Trains PyTorch models. Requires `dataenginex[pytorch]`.

## SentenceTransformerFinetuneTrainer

Fine-tunes sentence transformer embeddings.

## TrainingResult

```python
@dataclass
class TrainingResult:
    model_name: str
    version: str
    metrics: dict[str, float]
    parameters: dict[str, Any]
    duration_seconds: float
    artifact_path: str
    trained_at: datetime
```
