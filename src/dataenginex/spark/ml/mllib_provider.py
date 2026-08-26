"""MLlib provider (§20.9).

Wraps Spark MLlib operations with DEX governance.
Integrates with MLflow for experiment tracking and model registry.
"""

from __future__ import annotations

from typing import Any

import structlog

from dataenginex.foundation.ids import ProjectId

logger = structlog.get_logger()

__all__ = ["MLlibProvider"]

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    from pyspark.ml import Pipeline  # noqa: F401
    from pyspark.ml.classification import (
        DecisionTreeClassifier,
        GBTClassifier,
        LogisticRegression,
        RandomForestClassifier,
    )
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import (
        BinaryClassificationEvaluator,  # noqa: F401
        MulticlassClassificationEvaluator,  # noqa: F401
        RegressionEvaluator,  # noqa: F401
    )
    from pyspark.ml.feature import (
        StandardScaler,  # noqa: F401
        VectorAssembler,  # noqa: F401
    )
    from pyspark.ml.regression import (
        DecisionTreeRegressor,
        GBTRegressor,
        LinearRegression,
        RandomForestRegressor,
    )
    from pyspark.sql import SparkSession  # noqa: F401

    _PYSPARK_ML_AVAILABLE = True
except ImportError:
    _PYSPARK_ML_AVAILABLE = False


# Algorithm registry
CLASSIFICATION_ALGORITHMS = {
    "logistic_regression": "LogisticRegression",
    "decision_tree": "DecisionTreeClassifier",
    "random_forest": "RandomForestClassifier",
    "gbt": "GBTClassifier",
}

REGRESSION_ALGORITHMS = {
    "linear_regression": "LinearRegression",
    "decision_tree": "DecisionTreeRegressor",
    "random_forest": "RandomForestRegressor",
    "gbt": "GBTRegressor",
}

CLUSTERING_ALGORITHMS = {
    "kmeans": "KMeans",
}

# Lazy map — populated only when PySpark ML is available
_ALGO_CLASS_MAP: dict[str, type] = {}


def _build_algo_map() -> None:
    if not _PYSPARK_ML_AVAILABLE:
        return
    _ALGO_CLASS_MAP.update({
        "LogisticRegression": LogisticRegression,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GBTClassifier": GBTClassifier,
        "LinearRegression": LinearRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GBTRegressor": GBTRegressor,
        "KMeans": KMeans,
    })


_build_algo_map()


class MLlibProvider:
    """MLlib provider with DEX governance (§20.9).

    Provides:
    - Model training with governance
    - Experiment tracking via MLflow
    - Model registry integration
    - Feature engineering
    - Model evaluation
    """

    def __init__(
        self,
        project_id: ProjectId,
        mlflow_tracking_uri: str | None = None,
        spark_session: Any | None = None,
    ) -> None:
        self.project_id = project_id
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._spark = spark_session
        self._experiment_id: str | None = None

    def connect(self) -> None:
        """Initialize MLflow and Spark connections."""
        if _MLFLOW_AVAILABLE and self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(f"dex-{self.project_id}")
            self._experiment_id = mlflow.get_experiment_by_name(
                f"dex-{self.project_id}"
            ).experiment_id
            logger.info(
                "mlflow connected",
                tracking_uri=self.mlflow_tracking_uri,
                experiment_id=self._experiment_id,
            )

        if self._spark is None and _PYSPARK_ML_AVAILABLE:
            try:
                from pyspark.sql import SparkSession

                self._spark = SparkSession.builder.getOrCreate()
            except Exception as exc:
                logger.warning("could not get spark session", error=str(exc))

    def train(
        self,
        algorithm: str,
        dataset_ref: str,
        parameters: dict[str, Any] | None = None,
        feature_columns: list[str] | None = None,
        label_column: str = "label",
    ) -> dict[str, Any]:
        """Submit a training job with governance.

        Args:
            algorithm: Algorithm name (e.g., 'logistic_regression', 'random_forest')
            dataset_ref: Reference to training dataset (table name or path)
            parameters: Algorithm-specific hyperparameters
            feature_columns: List of feature column names
            label_column: Name of the label column

        Returns:
            Training result with model metrics and artifact paths
        """
        if not _PYSPARK_ML_AVAILABLE:
            return self._train_sklearn(algorithm, dataset_ref, parameters)

        if self._spark is None:
            msg = "No Spark session available"
            raise RuntimeError(msg)

        try:
            # Load dataset
            df = self._load_dataset(dataset_ref)

            # Prepare features
            if feature_columns is None:
                feature_columns = [c for c in df.columns if c != label_column]

            # Create ML pipeline
            pipeline = self._create_pipeline(
                algorithm, feature_columns, label_column, parameters or {}
            )

            # Split data
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

            # Train model
            with (
                mlflow.start_run(experiment_id=self._experiment_id)
                if _MLFLOW_AVAILABLE
                else self._no_op()
            ):
                model = pipeline.fit(train_df)

                # Evaluate model
                predictions = model.transform(test_df)
                metrics = self._evaluate_model(algorithm, predictions, label_column)

                # Log to MLflow
                if _MLFLOW_AVAILABLE:
                    mlflow.log_params(parameters or {})
                    mlflow.log_metrics(metrics)
                    mlflow.spark.log_model(model, "model")

                # Save model
                model_path = f"/user/hive/warehouse/models/{self.project_id}/{algorithm}"
                model.write().overwrite().save(model_path)

                logger.info(
                    "model trained",
                    algorithm=algorithm,
                    metrics=metrics,
                    model_path=model_path,
                )

                return {
                    "status": "completed",
                    "project_id": self.project_id,
                    "algorithm": algorithm,
                    "dataset_ref": dataset_ref,
                    "model_path": model_path,
                    "metrics": metrics,
                    "feature_columns": feature_columns,
                    "label_column": label_column,
                }

        except Exception as exc:
            logger.error("training failed", algorithm=algorithm, error=str(exc))
            return {
                "status": "error",
                "project_id": self.project_id,
                "algorithm": algorithm,
                "error": str(exc),
            }

    def _train_sklearn(
        self,
        algorithm: str,
        dataset_ref: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fallback training with scikit-learn (when PySpark ML not available)."""
        try:
            import mlflow.sklearn
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.metrics import accuracy_score, r2_score
            from sklearn.model_selection import train_test_split

            # Load dataset
            if dataset_ref.endswith(".csv"):
                df = pd.read_csv(dataset_ref)
            elif dataset_ref.endswith(".parquet"):
                df = pd.read_parquet(dataset_ref)
            else:
                # Try as table name
                df = pd.read_csv(dataset_ref)

            # Split features and label
            label_column = parameters.get("label_column", "label") if parameters else "label"
            feature_columns = [c for c in df.columns if c != label_column]

            X = df[feature_columns]
            y = df[label_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            with mlflow.start_run(experiment_id=self._experiment_id):
                if algorithm in ["logistic_regression", "logistic"]:
                    model = LogisticRegression(**(parameters or {}))
                elif algorithm in ["random_forest", "rf"]:
                    model = RandomForestClassifier(**(parameters or {}))
                elif algorithm in ["linear_regression", "linear"]:
                    model = LinearRegression(**(parameters or {}))
                else:
                    model = RandomForestRegressor(**(parameters or {}))

                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                if "classification" in algorithm or algorithm in ["logistic", "rf"]:
                    metrics = {"accuracy": accuracy_score(y_test, y_pred)}
                else:
                    metrics = {"r2": r2_score(y_test, y_pred)}

                # Log
                mlflow.log_params(parameters or {})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")

                return {
                    "status": "completed",
                    "project_id": self.project_id,
                    "algorithm": algorithm,
                    "metrics": metrics,
                    "framework": "sklearn",
                }

        except Exception as exc:
            logger.error("sklearn training failed", error=str(exc))
            return {
                "status": "error",
                "project_id": self.project_id,
                "algorithm": algorithm,
                "error": str(exc),
            }

    def _load_dataset(self, dataset_ref: str) -> Any:
        """Load dataset from reference.

        Args:
            dataset_ref: Table name, path, or data reference

        Returns:
            Spark DataFrame or pandas DataFrame
        """
        if self._spark:
            # Try as Delta table
            try:
                return self._spark.read.format("delta").load(dataset_ref)
            except Exception:
                pass

            # Try as Parquet
            try:
                return self._spark.read.parquet(dataset_ref)
            except Exception:
                pass

            # Try as CSV
            return self._spark.read.csv(dataset_ref, header=True, inferSchema=True)
        else:
            import pandas as pd

            if dataset_ref.endswith(".csv"):
                return pd.read_csv(dataset_ref)
            elif dataset_ref.endswith(".parquet"):
                return pd.read_parquet(dataset_ref)
            return pd.read_csv(dataset_ref)

    def _create_pipeline(
        self,
        algorithm: str,
        feature_columns: list[str],
        label_column: str,
        parameters: dict[str, Any],
    ) -> Any:
        """Create ML pipeline with preprocessing and algorithm."""
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import VectorAssembler

        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features",
            handleInvalid="skip",
        )

        algo_name = (
            CLASSIFICATION_ALGORITHMS.get(algorithm)
            or REGRESSION_ALGORITHMS.get(algorithm)
            or CLUSTERING_ALGORITHMS.get(algorithm)
        )
        if algo_name is None:
            msg = f"Unknown algorithm: {algorithm}"
            raise ValueError(msg)

        algo_cls = _ALGO_CLASS_MAP.get(algo_name)
        if algo_cls is None:
            msg = f"Algorithm not implemented: {algo_name}"
            raise ValueError(msg)

        kwargs = {"featuresCol": "features", **parameters}
        if algo_name != "KMeans":
            kwargs["labelCol"] = label_column
        algo = algo_cls(**kwargs)

        return Pipeline(stages=[assembler, algo])

    def _evaluate_model(
        self,
        algorithm: str,
        predictions: Any,
        label_column: str,
    ) -> dict[str, Any]:
        """Evaluate trained model.

        Args:
            algorithm: Algorithm name
            predictions: Predictions DataFrame
            label_column: Label column name

        Returns:
            Evaluation metrics
        """
        from pyspark.ml.evaluation import (
            BinaryClassificationEvaluator,
            MulticlassClassificationEvaluator,
            RegressionEvaluator,
        )

        metrics = {}

        if algorithm in CLASSIFICATION_ALGORITHMS:
            # Classification metrics
            binary_eval = BinaryClassificationEvaluator(labelCol=label_column)
            metrics["auc_roc"] = binary_eval.evaluate(predictions)

            multi_eval = MulticlassClassificationEvaluator(labelCol=label_column)
            metrics["accuracy"] = multi_eval.evaluate(
                predictions, {multi_eval.metricName: "accuracy"}
            )
            metrics["f1"] = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
        elif algorithm in REGRESSION_ALGORITHMS:
            # Regression metrics
            reg_eval = RegressionEvaluator(labelCol=label_column)
            metrics["rmse"] = reg_eval.evaluate(predictions, {reg_eval.metricName: "rmse"})
            metrics["mae"] = reg_eval.evaluate(predictions, {reg_eval.metricName: "mae"})
            metrics["r2"] = reg_eval.evaluate(predictions, {reg_eval.metricName: "r2"})
        elif algorithm in CLUSTERING_ALGORITHMS:
            # Clustering metrics
            from pyspark.ml.evaluation import ClusteringEvaluator

            cluster_eval = ClusteringEvaluator()
            metrics["silhouette"] = cluster_eval.evaluate(predictions)

        return metrics

    def register_model(
        self,
        model_path: str,
        model_name: str,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Register model in MLflow model registry.

        Args:
            model_path: Path to trained model
            model_name: Name for the model
            model_version: Version string

        Returns:
            Registration result
        """
        if not _MLFLOW_AVAILABLE:
            return {"status": "error", "error": "MLflow not available"}

        try:
            # Register model
            model_uri = (
                f"runs:/{mlflow.active_run().info.run_id}/model"
                if mlflow.active_run()
                else model_path
            )
            result = mlflow.register_model(model_uri, model_name)

            logger.info(
                "model registered",
                model_name=model_name,
                version=result.version,
            )

            return {
                "status": "registered",
                "model_name": model_name,
                "version": result.version,
                "model_path": model_path,
            }
        except Exception as exc:
            logger.error("model registration failed", error=str(exc))
            return {"status": "error", "error": str(exc)}

    def load_model(self, model_name: str, version: str | None = None) -> Any:
        """Load model from MLflow model registry.

        Args:
            model_name: Model name
            version: Optional version (loads latest if not specified)

        Returns:
            Loaded model
        """
        if not _MLFLOW_AVAILABLE:
            msg = "MLflow not available"
            raise RuntimeError(msg)

        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"

            return (
                mlflow.spark.load_model(model_uri)
                if self._spark
                else mlflow.pyfunc.load_model(model_uri)
            )
        except Exception as exc:
            logger.error("model loading failed", model_name=model_name, error=str(exc))
            raise

    def _no_op(self) -> Any:
        """Context manager that does nothing (for when MLflow is not available)."""
        from contextlib import contextmanager

        @contextmanager
        def noop() -> Any:
            yield None

        return noop()
