"""Resource estimation and strategy selection (§15.5, §15.6).

Two jobs, in order: predict what an operation will cost, then choose how to run
it within the budget that leaves.

The estimator is empirical where it can be and static where it cannot. A first
run has no history, so it uses the operation's declared request and says so —
``basis="static"``, low confidence. Once attempts have been observed it uses
those instead, and confidence rises with agreement between them rather than
with their number: ten observations that disagree wildly are not evidence of
anything, and treating them as ten times better than one would be exactly the
false precision §15.5 warns against.

The uncertainty band is the part that matters. An estimate of "2.3-3.1 GB,
confidence medium" is useful to a scheduler; "2.7 GB" is a number that will be
wrong and gives nothing to be careful with. Admission control uses the *upper*
bound, because being wrong about the ceiling is what causes an out-of-memory
kill.

Strategy selection (§15.6) is deliberately a small ladder rather than a cost
model. In-memory when it fits, chunked when it does not, spilling when even
chunks do not, sampling when nothing does. The chosen strategy is recorded so
provenance can answer "why did this run take that long?" — a run that silently
downgraded to sampling and produced different numbers is otherwise
indistinguishable from one that did not.
"""

from __future__ import annotations

import statistics
from enum import StrEnum

from dataenginex.foundation import (
    EstimateContext,
    Operation,
    ResourceEstimate,
    ResourceRequest,
)
from dataenginex.foundation.projects import FrozenModel

__all__ = [
    "EstimateBand",
    "ExecutionStrategy",
    "StrategyChoice",
    "choose_strategy",
    "estimate_operation",
]


class ExecutionStrategy(StrEnum):
    """How the planner decided to run something (§15.6)."""

    IN_MEMORY = "in_memory"
    CHUNKED = "chunked"
    SPILLING = "spilling"
    SAMPLED = "sampled"
    DEFERRED = "deferred"


class EstimateBand(FrozenModel):
    """A range, not a number (§15.5).

    Both bounds are reported because they answer different questions. The lower
    bound is what a user waits for at best; the upper bound is what the
    scheduler must actually reserve, since exceeding it is what gets a process
    killed.
    """

    low: float
    high: float

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2

    def describe(self, unit: str) -> str:
        """Human-readable, in the form §15.5 asks for: "2.3-3.1 GB"."""
        return f"{self.low:.1f}-{self.high:.1f} {unit}"


class StrategyChoice(FrozenModel):
    """The chosen strategy and why (§15.6).

    ``reason`` exists so the record is auditable. "Chose sampling" is not a
    finding; "chose sampling because the estimate exceeded the project's memory
    limit even when chunked" is.
    """

    strategy: ExecutionStrategy
    reason: str
    safe_under_limit: bool


def _confidence_from(samples: list[float]) -> float:
    """How much a set of observations of one quantity agree with each other.

    Spread, not count. Two consistent runs say more about the next one than ten
    that range over an order of magnitude, and rewarding volume alone would let
    a noisy operation look predictable just by running often.

    Computed per quantity rather than once for the whole history. An operation
    can be reliably fast and wildly variable in memory — a join whose build
    side depends on the data — and deriving the memory band from timing
    agreement would report a narrow band for exactly the case that gets a
    process OOM-killed.
    """
    if len(samples) < 2:
        # One observation is a data point, not a distribution. It beats nothing,
        # which is why this sits above the static floor and well below certainty.
        return 0.4 if samples else 0.2

    mean = statistics.fmean(samples)
    if mean <= 0:
        return 0.4
    # Coefficient of variation: dispersion relative to size, so a one-second
    # spread on a two-second operation counts as noisy and the same spread on
    # an hour-long one does not.
    spread = statistics.pstdev(samples) / mean
    return max(0.2, min(0.95, 1.0 - spread))


def _band(centre: float, confidence: float) -> EstimateBand:
    """Widen a point prediction by how little we trust it.

    At confidence 1.0 the band collapses to the point; at 0.2 it spans roughly
    a factor of two either way. The band is the honest part of the estimate.
    """
    margin = centre * (1.0 - confidence)
    return EstimateBand(low=max(0.0, centre - margin), high=centre + margin)


def estimate_operation(
    operation: Operation, context: EstimateContext
) -> tuple[ResourceEstimate, EstimateBand, EstimateBand]:
    """Predict cost from history, falling back to the declaration (§15.5).

    Returns the estimate plus explicit duration and memory bands. The bands are
    separate from :class:`ResourceEstimate` because that type crosses layers and
    carries a single confidence scalar, while presentation and admission control
    both need the actual range.
    """
    request = operation.resource_request
    history = context.history

    durations = [o.duration_seconds for o in history if o.duration_seconds is not None]
    memories = [float(o.peak_memory_mb) for o in history if o.peak_memory_mb is not None]

    # Separately, because the two disagree in practice. A join can be reliably
    # fast and unpredictable in memory; one confidence figure would report the
    # memory band as narrow on exactly the runs that get OOM-killed.
    confidence = _confidence_from(durations)
    memory_confidence = _confidence_from(memories)

    if durations:
        # Median, not mean: one pathological run — a cold cache, a machine that
        # was busy — should not drag every future estimate with it.
        duration = statistics.median(durations)
        basis = "empirical"
    else:
        # No history. A tenth of the timeout is a guess and is labelled as one;
        # the timeout is the only cost signal the declaration actually carries.
        duration = float(request.timeout_seconds) / 10
        basis = "static"

    memory_mb = float(statistics.median(memories)) if memories else float(request.memory_mb)

    # History gathered at a different input size is weaker evidence. Rather
    # than scaling the number by a linear model that is right for a scan and
    # wrong for a join, this lowers confidence and lets the band widen.
    if context.input_size_bytes and history:
        confidence = max(0.2, confidence - 0.1)
        memory_confidence = max(0.2, memory_confidence - 0.1)

    estimate = ResourceEstimate(
        request=ResourceRequest(
            cpu_cores=request.cpu_cores,
            memory_mb=max(1, int(memory_mb)),
            disk_mb=request.disk_mb,
            gpu_count=request.gpu_count,
            timeout_seconds=request.timeout_seconds,
        ),
        estimated_duration_seconds=duration,
        confidence=confidence,
        basis=basis,
    )
    return estimate, _band(duration, confidence), _band(memory_mb, memory_confidence)


def choose_strategy(
    estimate: ResourceEstimate,
    memory_band: EstimateBand,
    *,
    available_memory_mb: float,
    project_limit_mb: float,
) -> StrategyChoice:
    """Pick an execution strategy that fits the budget (§15.6).

    Decided against the band's *upper* bound throughout. Planning against the
    midpoint would be right on average and out of memory half the time, and an
    OOM kill costs the whole run rather than the difference between two
    strategies.
    """
    ceiling = min(available_memory_mb, project_limit_mb)
    needed = memory_band.high

    if needed <= ceiling * 0.5:
        return StrategyChoice(
            strategy=ExecutionStrategy.IN_MEMORY,
            reason=(
                f"estimated peak {needed:.0f}MB fits well inside the "
                f"{ceiling:.0f}MB available"
            ),
            safe_under_limit=True,
        )

    if needed <= ceiling:
        # It fits, but not with room to spare, and the estimate is a range.
        # Chunking costs some throughput and removes the cliff.
        return StrategyChoice(
            strategy=ExecutionStrategy.CHUNKED,
            reason=(
                f"estimated peak {needed:.0f}MB approaches the {ceiling:.0f}MB "
                "available; chunking trades throughput for headroom"
            ),
            safe_under_limit=True,
        )

    if needed <= project_limit_mb * 4:
        # Too big for memory, small enough that spilling to disk is plausible
        # rather than an all-night operation.
        return StrategyChoice(
            strategy=ExecutionStrategy.SPILLING,
            reason=(
                f"estimated peak {needed:.0f}MB exceeds the {ceiling:.0f}MB "
                "available; spilling intermediate state to disk"
            ),
            safe_under_limit=True,
        )

    # Nothing fits. Sampling produces an answer about a subset, which is a
    # different answer — so it is reported as not safe under the limit, and the
    # caller decides rather than discovering it in the results.
    return StrategyChoice(
        strategy=ExecutionStrategy.SAMPLED,
        reason=(
            f"estimated peak {needed:.0f}MB is far beyond the "
            f"{project_limit_mb:.0f}MB project limit; only a sample can run here, "
            "and its results describe the sample"
        ),
        safe_under_limit=False,
    )
