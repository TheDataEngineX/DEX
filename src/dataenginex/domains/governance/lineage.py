"""Provenance graph and standards projections (§8.5-8.6).

One internal graph, two external projections. The graph is the source of truth
and covers data, models, prompts, tools, policies, approvals, and transmissions
in a single typed structure; OpenLineage and W3C PROV are *views* over it,
generated on demand.

That direction matters. Modelling directly in OpenLineage would force every
DEX-specific relation — approvals, policy decisions, prompt provenance — into
custom facets, and modelling directly in PROV would lose the operational detail
the scheduler and retention logic need. Keeping an internal graph and projecting
outward means neither standard constrains what can be recorded.

Edges are append-only. A graph that can be edited cannot answer "what actually
happened", which is the only question it exists to answer.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from dataenginex.foundation import (
    LineageEdge,
    LineageNodeType,
    LineageRelation,
    ProjectId,
    RevisionId,
)
from dataenginex.runtime.state import ControlStore

__all__ = ["LineageService", "OPENLINEAGE_PRODUCER", "PROV_NAMESPACE"]

OPENLINEAGE_PRODUCER = "https://github.com/TheDataEngineX/dataenginex"
PROV_NAMESPACE = "https://thedataenginex.org/prov#"

# OpenLineage models runs as start/complete events over jobs; PROV models the
# same facts as activities over entities. The relations below are the ones that
# carry a natural equivalent in each.
_PROV_RELATIONS = {
    LineageRelation.CONSUMED: "used",
    LineageRelation.PRODUCED: "wasGeneratedBy",
    LineageRelation.DERIVED_FROM: "wasDerivedFrom",
    LineageRelation.TRAINED_ON: "wasDerivedFrom",
    LineageRelation.EVALUATED_WITH: "used",
    LineageRelation.PROMPTED_BY: "wasInfluencedBy",
    LineageRelation.RETRIEVED_FROM: "used",
    LineageRelation.APPROVED_BY: "wasAssociatedWith",
    LineageRelation.TRANSMITTED_TO: "wasInfluencedBy",
    LineageRelation.SUPERSEDES: "wasRevisionOf",
    LineageRelation.INVALIDATES: "wasInvalidatedBy",
    LineageRelation.DELETED_BECAUSE_OF: "wasInvalidatedBy",
}

# Agents in PROV terms — things that act rather than things acted upon.
_AGENT_TYPES = frozenset({LineageNodeType.AGENT, LineageNodeType.OPERATION})


class LineageService:
    """Records and queries the provenance graph (§8.5).

    Every write goes through the control store's transaction so an edge and the
    state change that caused it commit together — a lineage graph assembled by
    a best-effort second write is a graph with holes exactly where a crash
    happened.
    """

    def __init__(self, store: ControlStore) -> None:
        self.store = store

    # --- recording ----------------------------------------------------------

    def record(self, edges: Sequence[LineageEdge]) -> int:
        """Append edges. Returns how many were written."""
        if not edges:
            return 0
        with self.store.transaction() as tx:
            tx.executemany(
                "INSERT INTO lineage_edges (edge_id, source_id, source_type, target_id, "
                "target_type, relation, project_id, revision_id, run_id, created_at, "
                "attributes_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        edge.edge_id,
                        edge.source_id,
                        edge.source_type.value,
                        edge.target_id,
                        edge.target_type.value,
                        edge.relation.value,
                        edge.project_id,
                        edge.revision_id,
                        edge.run_id,
                        edge.created_at.isoformat(),
                        json.dumps(edge.attributes),
                    )
                    for edge in edges
                ],
            )
        return len(edges)

    def record_run(
        self,
        *,
        run_id: str,
        project_id: ProjectId,
        revision_id: RevisionId,
        consumed: Sequence[str] = (),
        produced: Sequence[str] = (),
        attributes: dict[str, str] | None = None,
    ) -> tuple[LineageEdge, ...]:
        """Record what one run read and wrote.

        This is the call that fixes the empty-graph bug: inputs and outputs are
        linked to the run *and* to each other, so a derived artifact can be
        traced to its sources without replaying the run.
        """
        shared: dict[str, Any] = {
            "project_id": project_id,
            "revision_id": revision_id,
            "run_id": run_id,
            "attributes": attributes or {},
        }
        edges: list[LineageEdge] = []

        for source in consumed:
            edges.append(
                LineageEdge(
                    source_id=run_id,
                    source_type=LineageNodeType.RUN,
                    target_id=source,
                    target_type=LineageNodeType.RESOURCE,
                    relation=LineageRelation.CONSUMED,
                    **shared,
                )
            )

        for output in produced:
            edges.append(
                LineageEdge(
                    source_id=run_id,
                    source_type=LineageNodeType.RUN,
                    target_id=output,
                    target_type=LineageNodeType.ARTIFACT,
                    relation=LineageRelation.PRODUCED,
                    **shared,
                )
            )
            # Direct derivation edges as well. Walking run nodes to answer
            # "where did this come from" costs a join per hop and breaks
            # entirely once a run is pruned.
            for source in consumed:
                edges.append(
                    LineageEdge(
                        source_id=output,
                        source_type=LineageNodeType.ARTIFACT,
                        target_id=source,
                        target_type=LineageNodeType.RESOURCE,
                        relation=LineageRelation.DERIVED_FROM,
                        **shared,
                    )
                )

        self.record(edges)
        return tuple(edges)

    def record_transmission(
        self,
        *,
        source_id: str,
        destination: str,
        project_id: ProjectId,
        revision_id: RevisionId | None = None,
        run_id: str | None = None,
        policy_decision_id: str | None = None,
    ) -> LineageEdge:
        """Record that data left the installation (§9.7).

        Kept in lineage rather than only in the audit log because the question
        asked after an incident is "what data reached this host", which is a
        graph traversal, not a log search.
        """
        attributes = {"destination": destination}
        if policy_decision_id:
            attributes["policy_decision_id"] = policy_decision_id

        edge = LineageEdge(
            source_id=source_id,
            source_type=LineageNodeType.ARTIFACT,
            target_id=destination,
            target_type=LineageNodeType.RESOURCE,
            relation=LineageRelation.TRANSMITTED_TO,
            project_id=project_id,
            revision_id=revision_id,
            run_id=run_id,
            attributes=attributes,
        )
        self.record([edge])
        return edge

    # --- querying -----------------------------------------------------------

    def edges_for(
        self,
        node_id: str,
        *,
        direction: str = "both",
        relations: Iterable[LineageRelation] = (),
        project_id: ProjectId | None = None,
    ) -> tuple[LineageEdge, ...]:
        """Edges touching a node.

        ``direction`` is ``upstream`` (what this came from), ``downstream``
        (what came from this), or ``both``.

        *project_id* scopes the answer. Node ids are names like ``orders``,
        which two projects may both use, so anything serving one project must
        pass it: invariant 6 forbids reading across the boundary, and an
        unscoped match would leak that another project has a table by the same
        name.
        """
        clauses = {
            "upstream": "source_id = ?",
            "downstream": "target_id = ?",
            "both": "(source_id = ? OR target_id = ?)",
        }
        if direction not in clauses:
            raise ValueError(f"direction must be one of {sorted(clauses)}, got {direction!r}")

        params: list[Any] = [node_id, node_id] if direction == "both" else [node_id]
        sql = f"SELECT * FROM lineage_edges WHERE {clauses[direction]}"
        if project_id is not None:
            sql += " AND project_id = ?"
            params.append(project_id)

        wanted = tuple(relations)
        if wanted:
            sql += " AND relation IN (" + ",".join("?" * len(wanted)) + ")"
            params.extend(r.value for r in wanted)

        rows = self.store.query(sql + " ORDER BY created_at", params)
        return tuple(_row_to_edge(row) for row in rows)

    def project_edges(
        self, project_id: ProjectId, *, limit: int | None = None
    ) -> tuple[LineageEdge, ...]:
        """Every edge in one project's graph, oldest first.

        The whole-graph read a lineage view needs. ``to_prov`` ran this query
        inline for its own projection; having it here means a caller that wants
        the edges rather than a PROV document does not go through a document.
        """
        sql = "SELECT * FROM lineage_edges WHERE project_id = ? ORDER BY created_at"
        params: list[Any] = [project_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return tuple(_row_to_edge(row) for row in self.store.query(sql, params))

    def upstream(self, node_id: str, *, max_depth: int = 10) -> tuple[str, ...]:
        """Everything ``node_id`` derives from, breadth-first.

        Depth-bounded and cycle-guarded: provenance graphs should be acyclic,
        but a bug that introduces a cycle must not hang a retention sweep.
        """
        return self._walk(node_id, max_depth=max_depth, forward=True)

    def downstream(self, node_id: str, *, max_depth: int = 10) -> tuple[str, ...]:
        """Everything derived from ``node_id`` — the deletion impact set (§8.10)."""
        return self._walk(node_id, max_depth=max_depth, forward=False)

    def _walk(self, node_id: str, *, max_depth: int, forward: bool) -> tuple[str, ...]:
        seen: set[str] = {node_id}
        order: list[str] = []
        frontier = [node_id]

        for _ in range(max_depth):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            column, other = ("source_id", "target_id") if forward else ("target_id", "source_id")
            rows = self.store.query(
                f"SELECT {other} AS next_id FROM lineage_edges WHERE {column} IN ({placeholders})",
                frontier,
            )
            frontier = []
            for row in rows:
                candidate = str(row["next_id"])
                if candidate not in seen:
                    seen.add(candidate)
                    order.append(candidate)
                    frontier.append(candidate)

        return tuple(order)

    def deletion_impact(self, resource_id: str) -> tuple[str, ...]:
        """What would be invalidated by deleting a resource (§8.10).

        Deleting a source without knowing its derived outputs leaves data that
        silently outlives the deletion request it was supposed to honour.
        """
        return self.downstream(resource_id)

    # --- projections (§8.6) -------------------------------------------------

    def to_openlineage(self, run_id: str) -> dict[str, Any]:
        """Project one run as an OpenLineage RunEvent.

        DEX-specific facts that have no OpenLineage equivalent go in a custom
        ``dex`` facet rather than being dropped — a projection that loses the
        approval trail is not usable for compliance.
        """
        edges = tuple(
            _row_to_edge(row)
            for row in self.store.query(
                "SELECT * FROM lineage_edges WHERE run_id = ? ORDER BY created_at", (run_id,)
            )
        )
        if not edges:
            return {}

        first = edges[0]
        inputs = [
            {"namespace": "dex", "name": e.target_id}
            for e in edges
            if e.relation is LineageRelation.CONSUMED
        ]
        outputs = [
            {"namespace": "dex", "name": e.target_id}
            for e in edges
            if e.relation is LineageRelation.PRODUCED
        ]
        transmissions = [
            e.attributes.get("destination", e.target_id)
            for e in edges
            if e.relation is LineageRelation.TRANSMITTED_TO
        ]

        return {
            "eventType": "COMPLETE",
            "eventTime": max(e.created_at for e in edges).isoformat(),
            "producer": OPENLINEAGE_PRODUCER,
            "schemaURL": (
                "https://openlineage.io/spec/1-0-5/OpenLineage.json#/definitions/RunEvent"
            ),
            "run": {
                "runId": run_id,
                "facets": {
                    "dex": {
                        "_producer": OPENLINEAGE_PRODUCER,
                        "projectId": first.project_id,
                        "revisionId": first.revision_id,
                        "transmissions": transmissions,
                    }
                },
            },
            "job": {"namespace": "dex", "name": f"{first.project_id}.{run_id}"},
            "inputs": inputs,
            "outputs": outputs,
        }

    def to_prov(self, project_id: ProjectId) -> dict[str, Any]:
        """Project a project's graph as W3C PROV-JSON.

        Entities are the things (resources, artifacts, prompts, models),
        activities are the runs and attempts that acted on them, and agents are
        the operations and AI agents that did the acting.
        """
        edges = tuple(
            _row_to_edge(row)
            for row in self.store.query(
                "SELECT * FROM lineage_edges WHERE project_id = ? ORDER BY created_at",
                (project_id,),
            )
        )

        entities: dict[str, Any] = {}
        activities: dict[str, Any] = {}
        agents: dict[str, Any] = {}
        relations: dict[str, dict[str, Any]] = {}

        def classify(node_id: str, node_type: LineageNodeType) -> None:
            key = f"dex:{node_id}"
            if node_type in (LineageNodeType.RUN, LineageNodeType.ATTEMPT):
                activities.setdefault(key, {"prov:type": node_type.value})
            elif node_type in _AGENT_TYPES:
                agents.setdefault(key, {"prov:type": node_type.value})
            else:
                entities.setdefault(key, {"prov:type": node_type.value})

        for edge in edges:
            classify(edge.source_id, edge.source_type)
            classify(edge.target_id, edge.target_type)
            relations[f"dex:{edge.edge_id}"] = {
                "prov:type": _PROV_RELATIONS.get(edge.relation, "wasInfluencedBy"),
                "dex:relation": edge.relation.value,
                "dex:source": f"dex:{edge.source_id}",
                "dex:target": f"dex:{edge.target_id}",
                "dex:time": edge.created_at.isoformat(),
            }

        return {
            "prefix": {"dex": PROV_NAMESPACE, "prov": "http://www.w3.org/ns/prov#"},
            "entity": entities,
            "activity": activities,
            "agent": agents,
            "wasInfluencedBy": relations,
        }


def _row_to_edge(row: sqlite3.Row) -> LineageEdge:
    return LineageEdge(
        edge_id=row["edge_id"],
        source_id=row["source_id"],
        source_type=LineageNodeType(row["source_type"]),
        target_id=row["target_id"],
        target_type=LineageNodeType(row["target_type"]),
        relation=LineageRelation(row["relation"]),
        project_id=ProjectId(row["project_id"]),
        revision_id=RevisionId(row["revision_id"]) if row["revision_id"] else None,
        run_id=row["run_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        attributes=json.loads(row["attributes_json"]),
    )
