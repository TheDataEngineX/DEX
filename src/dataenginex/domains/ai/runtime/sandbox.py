"""EXPERIMENTAL — Subprocess-based sandbox for isolated code execution.

.. warning::

   This module is **experimental** and provides only **process-level**
   isolation. It is **not safe** for running adversarial untrusted code.
   Use a container (Docker, gVisor) or VM for stronger isolation. The
   API and defaults may change without notice.

Runs code in a child process with enforced:
  - Wall-clock timeout (subprocess ``timeout`` parameter)
  - Memory cap (``RLIMIT_AS`` via ``resource.setrlimit`` on Linux/macOS)
  - CPU time cap (``RLIMIT_CPU``)
  - Clean environment (no parent env vars unless explicitly allowed)
  - Execution in a dedicated temp directory (removed after the run)

No network isolation is applied.

Supported languages
-------------------
python      — ``sys.executable -c <code>`` (or via temp file)
bash        — ``/bin/bash -c <code>`` (if available)

Suppressing the experimental warning::

    import warnings
    from dataenginex.domains.ai.runtime.sandbox import ExperimentalFeatureWarning
    warnings.filterwarnings("ignore", category=ExperimentalFeatureWarning)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ExperimentalFeatureWarning",
    "Sandbox",
    "SandboxConfig",
    "SandboxResult",
    "SandboxTimeoutError",
    "UnsupportedLanguageError",
]


class ExperimentalFeatureWarning(UserWarning):
    """Raised once per process when an experimental DataEngineX feature is used."""


# Module-level flag — emit experimental warning at most once per process.
_experimental_warned = False


def _warn_experimental_once() -> None:
    """Emit the experimental warning at most once per Python process."""
    global _experimental_warned
    if _experimental_warned:
        return
    _experimental_warned = True
    warnings.warn(
        "dataenginex.domains.ai.runtime.sandbox is EXPERIMENTAL — process-level "
        "isolation only, not safe for adversarial untrusted code. Use a "
        "container or VM for stronger isolation. API may change.",
        ExperimentalFeatureWarning,
        stacklevel=3,
    )


_DEFAULT_TIMEOUT = 30
_DEFAULT_MEMORY_MB = 256
_DEFAULT_CPU_S = 30
_SUPPORTED_LANGUAGES = frozenset({"python", "bash"})


class SandboxTimeoutError(RuntimeError):
    """Raised when sandboxed code exceeds the wall-clock timeout."""


class UnsupportedLanguageError(ValueError):
    """Raised when the requested language is not supported."""


@dataclass(frozen=True)
class SandboxConfig:
    """Limits applied to sandboxed execution.

    Attributes:
        timeout_s: Wall-clock timeout in seconds.
        memory_mb: Virtual address space cap in megabytes (Linux/macOS).
        cpu_time_s: CPU time limit in seconds (Linux/macOS).
        allowed_env_vars: Env var names to pass through to the child process.
            Defaults to a minimal safe set.
    """

    timeout_s: int = _DEFAULT_TIMEOUT
    memory_mb: int = _DEFAULT_MEMORY_MB
    cpu_time_s: int = _DEFAULT_CPU_S
    allowed_env_vars: tuple[str, ...] = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP")


@dataclass
class SandboxResult:
    """Result of a sandboxed execution.

    Attributes:
        output: Captured stdout.
        error: Captured stderr, or ``None`` if empty.
        exit_code: Process exit code (0 = success).
        timed_out: True if the process was killed due to timeout.
        metadata: Extra info (language, duration, etc.).
    """

    output: str
    error: str | None
    exit_code: int
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Sandbox:
    """Subprocess-based isolated execution environment.

    Parameters
    ----------
    config:
        Resource limits and allowed env vars.  Uses safe defaults.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        _warn_experimental_once()
        self._config = config or SandboxConfig()

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
        memory_mb: int | None = None,
    ) -> SandboxResult:
        """Execute *code* in an isolated subprocess.

        Parameters
        ----------
        code:
            Source code to run.
        language:
            ``"python"`` or ``"bash"``.
        timeout:
            Override config wall-clock timeout (seconds).
        memory_mb:
            Override config memory cap (megabytes).

        Returns
        -------
        SandboxResult
            stdout, stderr, exit code, and timeout flag.

        Raises
        ------
        UnsupportedLanguageError
            When *language* is not in ``{"python", "bash"}``.
        """
        lang = language.lower().strip()
        if lang not in _SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Language {language!r} is not supported. Supported: {sorted(_SUPPORTED_LANGUAGES)}"
            )

        effective_timeout = timeout if timeout is not None else self._config.timeout_s
        effective_memory_mb = memory_mb if memory_mb is not None else self._config.memory_mb

        with tempfile.TemporaryDirectory(prefix="dex_sandbox_") as tmpdir:
            return self._run(code, lang, effective_timeout, effective_memory_mb, Path(tmpdir))

    def _run(
        self,
        code: str,
        language: str,
        timeout_s: int,
        memory_mb: int,
        workdir: Path,
    ) -> SandboxResult:
        """Write code to a temp file and execute it in a subprocess."""
        # Write code to a file so we avoid shell injection via -c
        suffix = ".py" if language == "python" else ".sh"
        code_file = workdir / f"code{suffix}"
        code_file.write_text(code, encoding="utf-8")

        cmd = _build_command(language, code_file)
        env = _build_env(self._config.allowed_env_vars)

        preexec = _make_preexec(memory_mb, self._config.cpu_time_s)

        import time  # noqa: PLC0415

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(workdir),
                env=env,
                preexec_fn=preexec,
            )
            duration = time.monotonic() - start
            return SandboxResult(
                output=proc.stdout,
                error=proc.stderr or None,
                exit_code=proc.returncode,
                timed_out=False,
                metadata={"language": language, "duration_s": round(duration, 3)},
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                output="",
                error=f"Execution timed out after {timeout_s}s",
                exit_code=-1,
                timed_out=True,
                metadata={"language": language, "timeout_s": timeout_s},
            )


def _build_command(language: str, code_file: Path) -> list[str]:
    """Build the subprocess command for the given language."""
    if language == "python":
        return [sys.executable, str(code_file)]
    # bash
    return ["/bin/bash", str(code_file)]


def _build_env(allowed: tuple[str, ...]) -> dict[str, str]:
    """Return a minimal env dict containing only the allowed vars."""
    return {k: v for k, v in os.environ.items() if k in allowed}


def _make_preexec(memory_mb: int, cpu_time_s: int) -> Any:
    """Return a preexec_fn that sets resource limits, or None on Windows."""
    try:
        import resource  # noqa: PLC0415

        def _limit() -> None:
            import contextlib  # noqa: PLC0415

            mb = memory_mb * 1024 * 1024
            # RLIMIT_AS is unsupported on some platforms (e.g. macOS/aarch64);
            # swallow errors so the subprocess still runs without limits.
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_AS, (mb, mb))
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_s, cpu_time_s))

        return _limit
    except ImportError:
        # Windows — resource module not available; skip limits
        return None
