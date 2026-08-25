"""Episodic memory — experience replay for task-based learning."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Episode(BaseModel):
    """A recorded agent episode — a full task execution with outcome."""

    task: str
    steps: list[dict[str, Any]]
    outcome: str
    reward: float
    timestamp: float


class EpisodicMemory:
    """Experience replay memory — stores and retrieves past episodes."""

    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def add_episode(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def recall_similar(self, task: str, top_k: int = 5) -> list[Episode]:
        task_lower = task.lower()
        scored: list[tuple[int, Episode]] = []
        for ep in self._episodes:
            score = sum(1 for word in task_lower.split() if word in ep.task.lower())
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def best_episodes(self, top_k: int = 5) -> list[Episode]:
        return sorted(self._episodes, key=lambda e: e.reward, reverse=True)[:top_k]
