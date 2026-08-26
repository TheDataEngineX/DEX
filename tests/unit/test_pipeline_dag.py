"""Tests for pipeline DAG resolution."""

from __future__ import annotations

import pytest

from dataenginex.domains.data.pipeline.dag import (
    build_dag,
    downstream_of,
    resolve_execution_order,
    root_pipelines,
    topological_order,
)


class TestDagResolver:
    def test_no_dependencies(self) -> None:
        pipelines = {"a": [], "b": [], "c": []}
        order = resolve_execution_order(pipelines)
        assert set(order) == {"a", "b", "c"}

    def test_linear_chain(self) -> None:
        pipelines = {"a": [], "b": ["a"], "c": ["b"]}
        order = resolve_execution_order(pipelines)
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond_dependency(self) -> None:
        pipelines = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
        order = resolve_execution_order(pipelines)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_raises(self) -> None:
        pipelines = {"a": ["b"], "b": ["a"]}
        with pytest.raises(ValueError, match="[Cc]ycle"):
            resolve_execution_order(pipelines)

    def test_missing_dependency_raises(self) -> None:
        pipelines = {"a": ["nonexistent"]}
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_execution_order(pipelines)

    def test_single_pipeline(self) -> None:
        pipelines = {"only": []}
        order = resolve_execution_order(pipelines)
        assert order == ["only"]

    def test_empty_graph(self) -> None:
        assert resolve_execution_order({}) == []


class TestBuildDag:
    def test_builds_adjacency(self) -> None:
        class _Pipe:
            def __init__(self, deps: list[str]) -> None:
                self.depends_on = deps

        pipelines = {"a": _Pipe([]), "b": _Pipe(["a"]), "c": _Pipe(["a", "b"])}
        dag = build_dag(pipelines)
        assert dag == {"a": [], "b": ["a"], "c": ["a", "b"]}

    def test_empty_pipelines(self) -> None:
        assert build_dag({}) == {}

    def test_no_depends_on(self) -> None:
        class _Pipe:
            pass

        pipelines = {"x": _Pipe(), "y": _Pipe()}
        dag = build_dag(pipelines)
        assert dag == {"x": [], "y": []}


class TestRootPipelines:
    def test_finds_roots(self) -> None:
        dag = {"a": [], "b": ["a"], "c": ["a", "b"]}
        assert set(root_pipelines(dag)) == {"a"}

    def test_all_roots(self) -> None:
        pipelines = {"a": [], "b": [], "c": []}
        assert set(root_pipelines(pipelines)) == {"a", "b", "c"}

    def test_empty(self) -> None:
        assert root_pipelines({}) == []


class TestDownstreamOf:
    def test_direct_downstream(self) -> None:
        dag = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
        ds = downstream_of("a", dag)
        assert set(ds) == {"b", "c"}

    def test_leaf_node(self) -> None:
        pipelines = {"a": [], "b": ["a"]}
        ds = downstream_of("b", pipelines)
        assert ds == []

    def test_nonexistent_node(self) -> None:
        pipelines = {"a": []}
        ds = downstream_of("z", pipelines)
        assert ds == []


class TestTopologicalOrder:
    def test_linear_chain(self) -> None:
        pipelines = {"a": [], "b": ["a"], "c": ["b"]}
        order = topological_order(pipelines)
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond(self) -> None:
        pipelines = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
        order = topological_order(pipelines)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")

    def test_independent(self) -> None:
        pipelines = {"x": [], "y": []}
        order = topological_order(pipelines)
        assert set(order) == {"x", "y"}
