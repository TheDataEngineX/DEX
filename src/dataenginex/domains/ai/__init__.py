"""AI domain — prompts, retrieval, tools, memory scopes, agent runtime (§5.3).

Model *routing* is mechanism and lives in ``providers/model/``; what a tool
call is permitted to do, and which memory scope an agent may read, is meaning
and lives here. §9.10 depends on that split holding: policy has to be
enforceable without knowing which vendor answered.
"""

from __future__ import annotations

__all__: list[str] = []
