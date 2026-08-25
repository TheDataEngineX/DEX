"""Stateful agent executor — runs tools, manages memory, and tracks iterations.

ReAct-style execution loop:
  1. Build a context prompt from system prompt, recent memory, and tool list.
  2. Call the LLM provider to get a response.
  3. Parse the response:
     - ``TOOL: <name>\\nARGS: <json>``  → execute the named tool, loop.
     - Anything else                    → treat as final answer, return.
  4. Repeat until a final answer or ``max_iterations`` is reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from dataenginex import _json
from dataenginex.domains.ai.memory.base import BaseMemory, MemoryEntry
from dataenginex.domains.ai.tools import ToolRegistry

if TYPE_CHECKING:
    from dataenginex.domains.ai.providers import BaseProvider

_TOOL_PREFIX = "TOOL:"
_ARGS_PREFIX = "ARGS:"


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""

    name: str
    model: str
    system_prompt: str = ""
    tools: list[str] = []
    memory_type: str = "short_term"
    max_iterations: int = 10


@dataclass
class AgentResponse:
    """Result of a full agent run."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    iterations: int = 0


@dataclass
class StepResult:
    """Result of a single agent step."""

    action: str  # "tool_call" | "respond"
    result: str
    reasoning: str


class AgentExecutor:
    """Stateful agent executor — runs tools, manages memory, tracks iterations.

    Parameters
    ----------
    config:
        Agent configuration (name, model, system_prompt, max_iterations…).
    tool_registry:
        Registry of callable tools the agent may invoke.
    memory:
        Memory store used to persist conversation context across steps.
    provider:
        LLM provider used to generate responses. Must implement
        :class:`~dataenginex.providers.model.router.BaseProvider`.
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        memory: BaseMemory,
        provider: BaseProvider,
    ) -> None:
        self.config = config
        self.tool_registry = tool_registry
        self.memory = memory
        self.provider = provider
        self._iteration = 0
        self._tool_calls: list[dict[str, Any]] = []

    def run(self, user_input: str) -> AgentResponse:
        """Execute a full agent run for the given *user_input*.

        Loops up to ``config.max_iterations`` steps, calling :meth:`step`
        each time.  Returns as soon as the provider emits a final answer.
        """
        self._iteration = 0
        self._tool_calls = []
        self.memory.add(MemoryEntry(content=user_input, role="user"))

        while self._iteration < self.config.max_iterations:
            step_result = self.step()
            self._iteration += 1
            if step_result.action == "respond":
                return AgentResponse(
                    content=step_result.result,
                    tool_calls=self._tool_calls,
                    iterations=self._iteration,
                )

        return AgentResponse(
            content="Max iterations reached without a final answer.",
            tool_calls=self._tool_calls,
            iterations=self._iteration,
        )

    def step(self) -> StepResult:
        """Execute a single reasoning/action step.

        Builds a context prompt from memory, calls the provider, and parses
        the response into a :class:`StepResult`.
        """
        recent = self.memory.recent(n=10)
        context = "\n".join(f"[{e.role}]: {e.content}" for e in recent)
        tools_info = ", ".join(self.tool_registry.list()) or "none"

        prompt = (
            f"{self.config.system_prompt}\n\n"
            f"Available tools: {tools_info}\n\n"
            f"Conversation:\n{context}\n\n"
            "To use a tool respond with:\n"
            "TOOL: <tool_name>\n"
            'ARGS: {"key": "value"}\n\n'
            "Otherwise write your final answer directly."
        )

        response_text = self.provider.generate(prompt)
        self.memory.add(MemoryEntry(content=response_text, role="assistant"))

        if response_text.startswith(_TOOL_PREFIX):
            return self._handle_tool_call(response_text)

        return StepResult(action="respond", result=response_text, reasoning=response_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_tool_call(self, response_text: str) -> StepResult:
        """Parse and execute a TOOL: … ARGS: … block."""
        lines = response_text.splitlines()
        tool_name = lines[0][len(_TOOL_PREFIX) :].strip()

        args: dict[str, Any] = {}
        for line in lines[1:]:
            if line.startswith(_ARGS_PREFIX):
                try:
                    args = _json.loads(line[len(_ARGS_PREFIX) :].strip())
                except ValueError:
                    args = {}
                break

        try:
            tool_result = str(self.tool_registry.call(tool_name, **args))
        except KeyError as exc:
            tool_result = f"Tool not found: {exc}"
        except Exception as exc:  # noqa: BLE001
            tool_result = f"Tool error: {exc}"

        record: dict[str, Any] = {"tool": tool_name, "args": args, "result": tool_result}
        self._tool_calls.append(record)
        self.memory.add(MemoryEntry(content=f"Tool result: {tool_result}", role="tool"))

        return StepResult(action="tool_call", result=tool_result, reasoning=response_text)
