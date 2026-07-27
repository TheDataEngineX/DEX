# dataenginex.core

Foundation layer — medallion architecture, quality gates, registries, exceptions, and shared schemas.

## Quick import

```python
from dataenginex.core import (
    MedallionArchitecture, DataLayer, LayerConfiguration,
    QualityGate, QualityResult, QualityDimension,
    BackendRegistry,
    DataEngineXError, PipelineError, ConfigError,
)
```

______________________________________________________________________

## Medallion Architecture

`dataenginex.core.medallion_architecture`

Defines the Bronze → Silver → Gold layer model and per-layer configuration.

::: dataenginex.core.medallion_architecture

**Key classes:** `MedallionArchitecture`, `DataLayer`, `LayerConfiguration`

```python
from dataenginex.core.medallion_architecture import MedallionArchitecture, DataLayer, LayerConfiguration

arch = MedallionArchitecture(
    bronze=LayerConfiguration(path="data/bronze"),
    silver=LayerConfiguration(path="data/silver", quality_threshold=0.95),
    gold=LayerConfiguration(path="data/gold", quality_threshold=0.99),
)
```

______________________________________________________________________

## Quality Gates

`dataenginex.core.quality`

Declarative data quality checks that run at layer promotion boundaries.

::: dataenginex.core.quality

**Key classes:** `QualityGate`, `QualityResult`, `QualityDimension`

```python
from dataenginex.core.quality import QualityGate, QualityDimension

gate = QualityGate(
    dimensions=[
        QualityDimension.COMPLETENESS,
        QualityDimension.UNIQUENESS,
    ]
)
result = gate.run(df)
assert result.passed
```

______________________________________________________________________

## Interfaces

`dataenginex.core.interfaces`

Abstract base classes for connectors, transforms, and storage backends. Implement these to extend DEX with custom components.

::: dataenginex.core.interfaces

**Key classes:** `BaseConnector`, `BaseTransform`, `BaseTracker`, `BaseVectorStore`, `BaseLLMProvider`

______________________________________________________________________

## Registry

`dataenginex.core.registry`

Generic component registry used by connectors, transforms, models, and agents.

::: dataenginex.core.registry

**Key class:** `BackendRegistry`

```python
from dataenginex.core.registry import BackendRegistry

registry: BackendRegistry[MyPlugin] = BackendRegistry("my_plugin")
registry.register("my_plugin", MyPlugin)
plugin = registry.get("my_plugin")
```

______________________________________________________________________

## Schemas

`dataenginex.core.schemas`

Shared Pydantic response models used across the engine, CLI, and API layers.

::: dataenginex.core.schemas

**Key models:** `HealthResponse`, `RootResponse`, `EchoRequest`, `EchoResponse`, `ErrorResponse`, `ComponentStatus`

______________________________________________________________________

## Validators

`dataenginex.core.validators`

Reusable Pydantic validators and field helpers for config schema enforcement.

::: dataenginex.core.validators

______________________________________________________________________

## Exceptions

`dataenginex.core.exceptions`

Typed exception hierarchy for pipeline, validation, and resource errors.

::: dataenginex.core.exceptions

**Key exceptions:** `DataEngineXError`, `ConfigError`, `PipelineError`, `RegistryError`

```python
from dataenginex.core.exceptions import PipelineError

try:
    engine.run_pipeline("missing")
except PipelineError as e:
    print(e)
```
