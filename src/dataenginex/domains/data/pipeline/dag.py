"""DAG resolver for pipeline execution order.

Resolves cross-pipeline dependencies defined by `depends_on` in config.
Uses Kahn's algorithm (topological sort) to produce execution order.
"""

from __future__ import annotations

from collections import deque
from typing import Any

__all__ = [
    "build_dag",
    "downstream_of",
    "resolve_execution_order",
    "root_pipelines",
    "topological_order",
]


def build_dag(pipelines: dict[str, Any]) -> dict[str, list[str]]:
    """Return {name: [dep, ...]} mapping, reading ``depends_on`` from each pipeline config."""
    return {name: list(getattr(cfg, "depends_on", None) or []) for name, cfg in pipelines.items()}


def root_pipelines(dag: dict[str, list[str]]) -> list[str]:
    """Return pipelines with no dependencies — suitable as cron-triggered roots."""
    return [name for name, deps in dag.items() if not deps]


def downstream_of(name: str, dag: dict[str, list[str]]) -> list[str]:
    """Return direct dependents of *name* — pipelines whose ``depends_on`` includes it."""
    return [n for n, deps in dag.items() if name in deps]


def topological_order(dag: dict[str, list[str]]) -> list[str]:
    """Topological sort (Kahn's algorithm). Raises ``ValueError`` on cycle."""
    remaining: dict[str, int] = {n: len(deps) for n, deps in dag.items()}
    reverse: dict[str, list[str]] = {n: [] for n in dag}
    for node, deps in dag.items():
        for d in deps:
            if d in reverse:
                reverse[d].append(node)
    queue = [n for n, cnt in remaining.items() if cnt == 0]
    result: list[str] = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for child in reverse.get(node, []):
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    if len(result) != len(dag):
        raise ValueError("cycle detected in pipeline dependency graph")
    return result


def _validate_deps(pipelines: dict[str, list[str]]) -> None:
    """Raise KeyError if any dependency references a non-existent pipeline."""
    for name, deps in pipelines.items():
        for dep in deps:
            if dep not in pipelines:
                msg = f"Pipeline '{name}' depends on '{dep}' which does not exist"
                raise KeyError(msg)


def _build_graph(
    pipelines: dict[str, list[str]],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Build in-degree map and adjacency list from dependency graph."""
    in_degree: dict[str, int] = {name: 0 for name in pipelines}
    adjacency: dict[str, list[str]] = {name: [] for name in pipelines}
    for name, deps in pipelines.items():
        for dep in deps:
            adjacency[dep].append(name)
            in_degree[name] += 1
    return in_degree, adjacency


def resolve_execution_order(
    pipelines: dict[str, list[str]],
) -> list[str]:
    """Resolve execution order from dependency graph.

    Args:
        pipelines: Mapping of pipeline_name -> list of dependency names.

    Returns:
        List of pipeline names in valid execution order.

    Raises:
        KeyError: If a dependency references a non-existent pipeline.
        ValueError: If there is a cycle in the dependency graph.
    """
    _validate_deps(pipelines)
    in_degree, adjacency = _build_graph(pipelines)

    queue: deque[str] = deque(name for name, degree in in_degree.items() if degree == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(pipelines):
        msg = "Cycle detected in pipeline dependencies"
        raise ValueError(msg)

    return order
