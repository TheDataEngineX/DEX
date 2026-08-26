"""DAG-based agent chaining — define and execute multi-agent workflows."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from dataenginex.domains.ai.runtime.executor import AgentConfig

if TYPE_CHECKING:
    from dataenginex.domains.ai.providers import BaseProvider


class AgentDAG:
    """Directed acyclic graph of agent nodes for multi-agent workflows.

    Each node is an :class:`~dataenginex.domains.ai.runtime.executor.AgentConfig`.
    Edges define execution order — output of the source node is passed as
    input to the destination node.

    Usage::

        dag = AgentDAG()
        dag.add_node("extract", AgentConfig(name="extract", model="gpt-4o"))
        dag.add_node("summarise", AgentConfig(name="summarise", model="claude-sonnet-4-6"))
        dag.add_edge("extract", "summarise")

        providers = {"extract": openai_provider, "summarise": anthropic_provider}
        results = dag.execute(providers=providers, initial_input="raw text here")
    """

    def __init__(self) -> None:
        self._nodes: dict[str, AgentConfig] = {}
        self._edges: list[tuple[str, str]] = []

    def add_node(self, agent_name: str, config: AgentConfig) -> None:
        """Add an agent node to the DAG."""
        self._nodes[agent_name] = config

    def add_edge(self, from_agent: str, to_agent: str) -> None:
        """Add a directed edge — output of *from_agent* feeds into *to_agent*."""
        self._edges.append((from_agent, to_agent))

    def validate(self) -> bool:
        """Return ``True`` if the DAG has no cycles, ``False`` otherwise."""
        visited: set[str] = set()
        in_stack: set[str] = set()
        adjacency: dict[str, list[str]] = {name: [] for name in self._nodes}
        for src, dst in self._edges:
            adjacency.setdefault(src, []).append(dst)

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if _has_cycle(neighbor):
                        return True
                elif neighbor in in_stack:
                    return True
            in_stack.discard(node)
            return False

        return all(not (node not in visited and _has_cycle(node)) for node in self._nodes)

    def execute(
        self,
        providers: dict[str, BaseProvider] | None = None,
        initial_input: str = "",
    ) -> dict[str, Any]:
        """Execute agents in topological order, chaining outputs as inputs.

        Parameters
        ----------
        providers:
            Map of *agent_name* → ``BaseProvider``.  If a node has no entry,
            its provider is skipped and a placeholder result is recorded.
        initial_input:
            Text passed to the first node(s) in the graph.

        Returns
        -------
        dict[str, Any]
            Map of *agent_name* → generated output string.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        if not self.validate():
            raise ValueError("DAG contains a cycle — cannot execute.")

        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = dict.fromkeys(self._nodes, 0)
        adjacency: dict[str, list[str]] = {name: [] for name in self._nodes}
        for src, dst in self._edges:
            adjacency[src].append(dst)
            in_degree[dst] += 1

        queue: deque[str] = deque(n for n in self._nodes if in_degree[n] == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        results: dict[str, Any] = {}
        current_input = initial_input

        for node_name in order:
            config = self._nodes[node_name]
            if providers and node_name in providers:
                prompt = current_input or config.system_prompt or ""
                output = providers[node_name].generate(prompt)
            else:
                output = f"[{node_name}: no provider registered]"
            results[node_name] = output
            current_input = output  # chain output → next node input

        return results
