# CLI Commands

The `dex` command group provides CLI access to DataEngineX.

```bash
dex validate dex.yaml     # Validate config
dex version               # Show version
dex run <pipeline>        # Queue a pipeline run
dex worker start          # Start a worker
dex secops status         # PrivacyGuard status
dex train <model>         # Train an ML model
```

## Commands

### `dex validate`

```bash
dex validate <config_path>
```

Compile a project config and report issues. Exits non-zero on validation errors.

### `dex version`

```bash
dex version
```

Show version, Python version, platform, and engine status.

### `dex run`

```bash
dex run <pipeline> --config <path> [--state-dir <path>]
dex run --all --config <path>
dex run <pipeline> --dry-run --config <path>
```

Queue workloads for a worker to execute. Nothing runs until a worker claims the run.

### `dex worker`

```bash
dex worker start [--state-dir <path>]
dex worker list [--state-dir <path>]
```

Start a worker process that claims and executes queued runs.

### `dex runtime`

```bash
dex runtime serve [--state-dir <path>]
dex runtime tick [--state-dir <path>]
dex runtime schedules [--state-dir <path>]
```

Control-plane daemon: fire scheduled runs, reclaim stale leases.

### `dex secops`

```bash
dex secops status [--config <path>]
dex secops scan [--config <path>] [--text <string>]
```

PrivacyGuard inspection and PII scanning.

### `dex train`

```bash
dex train <model_name> --config <path>
```

Train an ML model from config.

### `dex studio`

```bash
dex studio [--host <addr>] [--port <port>] [--reload]
```

Serve the DEX Studio web UI.
