"""Spark Connect client and server management (§20.4).

Spark Connect is the preferred managed execution boundary separating
the DEX control plane from Spark execution environments.
"""

from __future__ import annotations

from dataenginex.spark.connect.client import SparkConnectClient
from dataenginex.spark.connect.server_manager import SparkServerManager
from dataenginex.spark.connect.session_registry import SparkSessionRegistry

__all__ = ["SparkConnectClient", "SparkServerManager", "SparkSessionRegistry"]
