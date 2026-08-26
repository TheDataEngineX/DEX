"""Resource estimation and strategy selection (§15.5, §15.6).

The claim under test is not "the estimate is accurate" — it cannot be, and
§15.5 says as much. It is that the estimate is *honest about how accurate it
is*, and that the planner behaves conservatively when it is not sure. Those
are testable; precision is not.
"""

from __future__ import annotations

from dataenginex.foundation import (
    EstimateContext,
    ObservedResources,
    Operation,
    ProjectId,
    ResourceEstimate,
    ResourceRequest,
    RevisionId,
)
from dataenginex.runtime.planning import (
    EstimateBand,
    ExecutionStrategy,
    choose_strategy,
    estimate_operation,
)

PROJECT = ProjectId("proj_plan")
REVISION = RevisionId("rev_1")


def make_operation() -> Operation:
    return Operation(
        operation_type="transform",
        resource_request=ResourceRequest(memory_mb=1024, timeout_seconds=600),
    )


def context(
    *observations: ObservedResources, input_size_bytes: int | None = None
) -> EstimateContext:
    return EstimateContext(
        project_id=PROJECT,
        revision_id=REVISION,
        history=observations,
        input_size_bytes=input_size_bytes,
    )


def observed(duration: float, memory: int) -> ObservedResources:
    return ObservedResources(duration_seconds=duration, peak_memory_mb=memory)


def estimate_with_memory(peak_mb: int) -> tuple[ResourceEstimate, EstimateBand]:
    """An estimate whose memory band centres on ``peak_mb``.

    Two identical observations give high confidence and therefore a narrow
    band, which keeps the strategy tests about the ladder rather than about how
    wide the uncertainty happens to be.
    """
    estimate, _duration, memory = estimate_operation(
        make_operation(), context(observed(10.0, peak_mb), observed(10.0, peak_mb))
    )
    return estimate, memory


# --- estimating (§15.5) -----------------------------------------------------


def test_with_no_history_the_estimate_says_it_is_a_guess() -> None:
    estimate, duration, _memory = estimate_operation(make_operation(), context())

    assert estimate.basis == "static"
    assert estimate.confidence <= 0.2
    # A guess with a wide band is useful; a guess presented as a number is not.
    assert duration.low < duration.high


def test_history_replaces_the_declaration() -> None:
    """The declared request is what the author thought. Observations are what
    happened, and they win."""
    estimate, _duration, memory = estimate_operation(
        make_operation(),
        context(observed(10.0, 200), observed(11.0, 210), observed(10.5, 205)),
    )

    assert estimate.basis == "empirical"
    assert 190 <= memory.midpoint <= 220, "still using the declared 1024MB"


def test_agreeing_observations_raise_confidence() -> None:
    consistent = context(observed(10.0, 200), observed(10.1, 201), observed(9.9, 199))
    noisy = context(observed(1.0, 200), observed(60.0, 200), observed(300.0, 200))

    steady, _, _ = estimate_operation(make_operation(), consistent)
    erratic, _, _ = estimate_operation(make_operation(), noisy)

    assert steady.confidence > erratic.confidence


def test_many_disagreeing_observations_are_not_evidence() -> None:
    """Confidence comes from agreement, not volume. Rewarding count alone
    would let a noisy operation look predictable just by running often."""
    few = context(observed(10.0, 100), observed(10.0, 100))
    many = context(*[observed(d, 100) for d in (1.0, 50.0, 2.0, 90.0, 5.0, 120.0)])

    steady, _, _ = estimate_operation(make_operation(), few)
    erratic, _, _ = estimate_operation(make_operation(), many)

    assert steady.confidence > erratic.confidence


def test_one_pathological_run_does_not_poison_the_estimate() -> None:
    """Median, not mean. A cold cache or a busy machine produces one outlier
    that a mean would carry into every future estimate."""
    _estimate, duration, _memory = estimate_operation(
        make_operation(),
        context(observed(10.0, 100), observed(10.0, 100), observed(600.0, 100)),
    )

    assert duration.midpoint < 100


def test_lower_confidence_widens_the_band() -> None:
    """The band is the honest part. A prediction nobody trusts must not look
    the same as one that has been right ten times."""
    steady, steady_band, _ = estimate_operation(
        make_operation(), context(observed(10.0, 100), observed(10.0, 100))
    )
    erratic, erratic_band, _ = estimate_operation(
        make_operation(), context(observed(2.0, 100), observed(40.0, 100))
    )

    assert steady.confidence > erratic.confidence
    steady_width = (steady_band.high - steady_band.low) / steady_band.midpoint
    erratic_width = (erratic_band.high - erratic_band.low) / erratic_band.midpoint
    assert erratic_width > steady_width


def test_stable_timing_does_not_imply_stable_memory() -> None:
    """Confidence is computed per quantity.

    A join can be reliably fast and unpredictable in memory — its build side
    depends on the data. Deriving the memory band from timing agreement would
    report a narrow band on exactly the runs that get OOM-killed, which is how
    the first version of this estimator was wrong.
    """
    _estimate, duration, memory = estimate_operation(
        make_operation(),
        context(observed(10.0, 400), observed(10.0, 1200), observed(10.0, 800)),
    )

    duration_width = (duration.high - duration.low) / duration.midpoint
    memory_width = (memory.high - memory.low) / memory.midpoint
    assert memory_width > duration_width


def test_a_different_input_size_lowers_confidence() -> None:
    """History gathered on other inputs is weaker evidence. Scaling it by a
    linear model would be right for a scan and wrong for a join, so the
    estimator widens the band instead of inventing a better number."""
    history = (observed(10.0, 100), observed(10.0, 100))
    same, _, _ = estimate_operation(make_operation(), context(*history))
    resized, _, _ = estimate_operation(
        make_operation(), context(*history, input_size_bytes=5_000_000_000)
    )

    assert resized.confidence < same.confidence


def test_a_band_reads_the_way_the_spec_asks() -> None:
    """§15.5's example output is "2.3-3.1 GB". A band that cannot be shown to a
    user is only half a feature."""
    assert EstimateBand(low=2.3, high=3.1).describe("GB") == "2.3-3.1 GB"


# --- choosing a strategy (§15.6) -------------------------------------------


def test_small_work_runs_in_memory() -> None:
    estimate, band = estimate_with_memory(100)
    choice = choose_strategy(
        estimate, band, available_memory_mb=8000, project_limit_mb=8000
    )

    assert choice.strategy is ExecutionStrategy.IN_MEMORY
    assert choice.safe_under_limit


def test_work_that_barely_fits_is_chunked() -> None:
    """It fits, but not with room to spare, and the estimate is a range.
    Chunking costs throughput and removes the cliff."""
    estimate, band = estimate_with_memory(900)
    choice = choose_strategy(
        estimate, band, available_memory_mb=1000, project_limit_mb=1000
    )

    assert choice.strategy is ExecutionStrategy.CHUNKED
    assert choice.safe_under_limit


def test_work_that_does_not_fit_spills() -> None:
    estimate, band = estimate_with_memory(3000)
    choice = choose_strategy(
        estimate, band, available_memory_mb=1000, project_limit_mb=1000
    )

    assert choice.strategy is ExecutionStrategy.SPILLING


def test_work_that_cannot_run_says_so() -> None:
    """Sampling answers a different question. Reporting it as safe would let a
    run quietly return numbers about a subset."""
    estimate, band = estimate_with_memory(100_000)
    choice = choose_strategy(
        estimate, band, available_memory_mb=1000, project_limit_mb=1000
    )

    assert choice.strategy is ExecutionStrategy.SAMPLED
    assert not choice.safe_under_limit


def test_the_decision_uses_the_upper_bound() -> None:
    """Planning against the midpoint would be right on average and out of
    memory half the time. An OOM kill costs the whole run."""
    estimate, _duration, band = estimate_operation(
        make_operation(), context(observed(10.0, 400), observed(10.0, 1200))
    )
    assert band.midpoint < 1000 < band.high, "the fixture no longer straddles the limit"

    choice = choose_strategy(
        estimate, band, available_memory_mb=1000, project_limit_mb=1000
    )
    assert choice.strategy is not ExecutionStrategy.IN_MEMORY


def test_every_choice_explains_itself() -> None:
    """§15.6 requires the strategy be recorded in provenance. "Chose sampling"
    is not a finding; the reason is what makes a slow run explainable later."""
    for needed, available in ((100, 8000), (900, 1000), (3000, 1000), (100_000, 1000)):
        estimate, band = estimate_with_memory(needed)
        choice = choose_strategy(
            estimate, band, available_memory_mb=available, project_limit_mb=available
        )
        assert "MB" in choice.reason
        assert len(choice.reason) > 20


def test_the_tightest_constraint_wins() -> None:
    """A machine with plenty of RAM does not entitle a project to exceed the
    limit it declared."""
    estimate, band = estimate_with_memory(900)
    choice = choose_strategy(
        estimate, band, available_memory_mb=64_000, project_limit_mb=1000
    )

    assert choice.strategy is ExecutionStrategy.CHUNKED
