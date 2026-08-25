"""Pydantic-Settings based environment configuration for DataEngineX.

Typed, validated settings for known variables (API keys, feature flags, etc.)
sourced from environment variables and ``.env`` files.

Any variable returned by :meth:`DexSettings.as_env_dict` is available for
``${VAR}`` interpolation in ``dex.yaml`` via
:func:`~dataenginex.config.loader.load_config`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_files(config_path: Path) -> list[Path]:
    """Return deduplicated .env file candidates for the given config path."""
    found: list[Path] = []
    seen: set[Path] = set()
    for candidate in [config_path.parent / ".env", Path.cwd() / ".env"]:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            found.append(candidate)
    return found


class DexSettings(BaseSettings):
    """Runtime settings for DataEngineX.

    Loaded from environment variables and .env files via pydantic-settings.
    Declared fields provide typed, validated access to known settings.
    All values (plus ``os.environ``) are accessible via :meth:`as_env_dict`
    for ``${VAR}`` substitution in ``dex.yaml``.

    Priority (highest first):
    1. Environment variables
    2. Project-level ``.env`` (same directory as ``dex.yaml``)
    3. CWD-level ``.env``
    4. Declared field defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False,
    )

    # ── Known typed fields ────────────────────────────────────────────────────
    # Add any well-known env vars here for typed access + IDE autocompletion.
    # Unknown vars in .env files are still picked up by as_env_dict().
    tmdb_api_key: str = ""

    # Track which .env files were loaded (set by for_config)
    _loaded_env_files: list[Path] = PrivateAttr(default_factory=list)

    @classmethod
    def for_config(cls, config_path: Path) -> DexSettings:
        """Instantiate settings scoped to the given config file's directory.

        Loads .env files from:
        - ``config_path.parent/.env``  (project secrets)
        - ``CWD/.env``                 (workspace secrets)
        """
        env_files = _find_env_files(config_path)
        env_file_tuple = tuple(env_files) if env_files else (".env",)

        # Create a dynamic subclass with custom env_file
        class DynamicSettings(cls):  # type: ignore[valid-type]
            model_config = SettingsConfigDict(
                env_file=env_file_tuple,
                env_file_encoding="utf-8",
                extra="allow",
                case_sensitive=False,
            )

        inst = DynamicSettings()
        inst._loaded_env_files = env_files  # noqa: SLF001
        return inst

    def _raw_env_file_vars(self) -> dict[str, str]:
        """Layer 1: raw values from loaded .env files (undeclared vars)."""
        result: dict[str, str] = {}
        try:
            from dotenv import dotenv_values

            for path in self._loaded_env_files:
                for k, v in dotenv_values(path).items():
                    if v is not None:
                        result[k] = v
        except ImportError:
            pass
        return result

    def as_env_dict(self) -> dict[str, str]:
        """Return a merged env dict for ``${VAR}`` resolution in dex.yaml.

        Builds the dict in priority order (lowest → highest, later writes win):

        1. Raw values from each loaded ``.env`` file (arbitrary undeclared keys)
        2. Declared field values resolved by pydantic-settings
        3. ``os.environ`` (always wins — never clobbered)
        """
        result = self._raw_env_file_vars()

        # Layer 2 — declared fields resolved by pydantic-settings
        for key, val in self.model_dump().items():
            if val:
                result[key.upper()] = str(val)
        for key, val in (self.model_extra or {}).items():
            if val is not None:
                result[key.upper()] = str(val)

        # Layer 3 — os.environ always wins
        result.update(os.environ)

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a setting by env-var name (case-insensitive)."""
        return self.as_env_dict().get(key, default)
