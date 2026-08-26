"""Deployment settings — the knobs a profile turns (§11).

Everything here has a working default, because the Lite profile has to start
with no configuration at all (§11.3). What a profile changes is *where* state
lives and *how much* runs at once, not which code paths exist.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Settings"]

DEFAULT_STATE_DIR = ".dex"
DEFAULT_CONTROL_DB = "control.db"


@dataclass(frozen=True)
class Settings:
    """Where a deployment keeps its state and how hard it is allowed to work.

    Frozen because settings are read during assembly and then relied upon.
    A value that could change after the store was opened would describe a
    deployment that no longer exists.
    """

    state_dir: Path = Path(DEFAULT_STATE_DIR)
    control_db_name: str = DEFAULT_CONTROL_DB
    #: SQLite busy timeout. Generous by default: a lock held by a scheduler
    #: tick is normal and short, and failing fast on it would surface as a
    #: spurious error rather than the brief wait it actually is.
    store_timeout_seconds: float = 30.0

    @property
    def control_db_path(self) -> Path:
        return self.state_dir / self.control_db_name

    @classmethod
    def from_env(cls, *, state_dir: Path | str | None = None) -> Settings:
        """Build settings from the environment, with an explicit override.

        The argument wins over ``DEX_STATE_DIR`` so a caller that already knows
        where the project lives — the CLI resolving a project directory, a test
        using a temp dir — is not overridden by an ambient variable it did not
        set.
        """
        resolved = state_dir if state_dir is not None else os.environ.get("DEX_STATE_DIR")
        return cls(state_dir=Path(resolved) if resolved is not None else Path(DEFAULT_STATE_DIR))
