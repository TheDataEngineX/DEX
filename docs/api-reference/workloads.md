# Workloads

Workload types, run states, and execution definitions.

```python
from dataenginex.foundation.workloads import WorkloadKind, RunState
```

## WorkloadKind

```python
class WorkloadKind(StrEnum):
    INTERACTIVE = "interactive"
    BATCH = "batch"
    SPARK_STREAM = "spark_stream"
    SPARK_PIPELINE = "spark_pipeline"
    EXTERNAL_ACTION = "external_action"
    SERVICE = "service"
```

## RunState

```python
class RunState(StrEnum):
    REQUESTED = "requested"
    AWAITING_POLICY = "awaiting_policy"
    AWAITING_APPROVAL = "awaiting_approval"
    PLANNING = "planning"
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
```

## Data Classes

| Class | Description |
|-------|-------------|
| `WorkloadDefinition` | Full workload spec (kind, config, retry, resources) |
| `WorkloadRun` | A single run of a workload |
| `ExecutionAttempt` | One attempt within a run |
| `InteractiveRequest` | Interactive workload request |
| `InteractiveResult` | Interactive workload result |
| `ObservedResources` | Resource usage observation |

## Functions

| Function | Description |
|----------|-------------|
| `can_transition(from_state, to_state)` | Check if a run state transition is valid |
| `can_service_transition(from_state, to_state)` | Check service state transitions |
| `is_terminal(state)` | Check if a run state is terminal |
| `is_service_terminal(state)` | Check if a service state is terminal |
