"""Spark MLlib and MLflow integration (§20.9).

DEX adds trust, governance, lineage, promotion, auditability — not
feature engineering or training abstractions.
"""

from dataenginex.spark.ml.mllib_provider import MLlibProvider
from dataenginex.spark.ml.model_contract import ModelContract

__all__ = ["MLlibProvider", "ModelContract"]
