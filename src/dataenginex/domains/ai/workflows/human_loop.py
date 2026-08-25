"""Human-in-the-loop approval gates for agent workflows."""

from __future__ import annotations

from typing import Any


class ApprovalGate:
    """Gate that pauses workflow execution until a human approves.

    Reads from stdin via :func:`input` — suitable for CLI and notebook contexts.
    In headless / automated environments, use :meth:`request_approval` with a
    pre-wired callback by subclassing and overriding :meth:`_prompt`.

    Parameters
    ----------
    description:
        Human-readable description of what is being approved.
    timeout_seconds:
        Advisory timeout shown to the operator (not enforced — ``input()``
        blocks indefinitely).  Subclass and override :meth:`_prompt` to add
        real timeout behaviour.
    """

    def __init__(self, description: str, timeout_seconds: int = 300) -> None:
        self.description = description
        self.timeout_seconds = timeout_seconds

    def request_approval(self, context: dict[str, Any]) -> bool:
        """Ask a human operator to approve or reject the pending action.

        Prints *description* and *context* to stdout and waits for ``y``/``n``
        input.  Returns ``True`` on approval, ``False`` otherwise (including
        EOF / keyboard interrupt).
        """
        return self._prompt(context)

    def _prompt(self, context: dict[str, Any]) -> bool:
        """Display the approval request and collect the operator's answer."""
        print(f"\n[ApprovalGate] {self.description}")
        print(f"Context   : {context}")
        print(f"Timeout   : {self.timeout_seconds}s (advisory)")
        try:
            answer = input("Approve? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")
