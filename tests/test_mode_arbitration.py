import pytest

from src.ins.mode_arbitration import GnssModeArbitrator, NavigationMode


def test_valid_gnss_keeps_aided_mode() -> None:
    arbitrator = GnssModeArbitrator(transition_duration_s=2.0)

    decision = arbitrator.update(10.0, True)

    assert decision.mode is NavigationMode.GNSS_AIDED
    assert decision.gnss_blend_weight == 1.0
    assert decision.fix_accepted is True


@pytest.mark.parametrize("fix_accepted", [False, None])
def test_first_rejected_or_missing_fix_immediately_enters_dead_reckoning(
    fix_accepted: bool | None,
) -> None:
    arbitrator = GnssModeArbitrator(transition_duration_s=2.0)

    decision = arbitrator.update(10.0, fix_accepted)

    assert decision.mode is NavigationMode.DEAD_RECKONING
    assert decision.gnss_blend_weight == 0.0
    assert decision.fix_accepted is False


def test_reacquisition_blends_monotonically_before_returning_to_aided() -> None:
    arbitrator = GnssModeArbitrator(transition_duration_s=4.0)
    arbitrator.update(10.0, False)

    decisions = [
        arbitrator.update(20.0, True),
        arbitrator.update(21.0, True),
        arbitrator.update(22.0, True),
        arbitrator.update(23.0, True),
        arbitrator.update(24.0, True),
    ]

    assert [decision.mode for decision in decisions] == [
        NavigationMode.TRANSITIONING,
        NavigationMode.TRANSITIONING,
        NavigationMode.TRANSITIONING,
        NavigationMode.TRANSITIONING,
        NavigationMode.GNSS_AIDED,
    ]
    weights = [decision.gnss_blend_weight for decision in decisions]
    assert weights == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert weights == sorted(weights)


def test_bad_fix_during_transition_aborts_blend_immediately() -> None:
    arbitrator = GnssModeArbitrator(transition_duration_s=4.0)
    arbitrator.update(0.0, False)
    arbitrator.update(1.0, True)
    assert arbitrator.update(2.0, True).gnss_blend_weight == pytest.approx(0.25)

    rejected = arbitrator.update(2.1, False)
    reacquired = arbitrator.update(5.0, True)

    assert rejected.mode is NavigationMode.DEAD_RECKONING
    assert rejected.gnss_blend_weight == 0.0
    assert reacquired.mode is NavigationMode.TRANSITIONING
    assert reacquired.gnss_blend_weight == 0.0


def test_repeated_timestamp_does_not_advance_blend() -> None:
    arbitrator = GnssModeArbitrator(transition_duration_s=2.0)
    arbitrator.update(0.0, False)
    first = arbitrator.update(1.0, True)
    repeated = arbitrator.update(1.0, True)

    assert first.gnss_blend_weight == 0.0
    assert repeated.gnss_blend_weight == 0.0
    assert repeated.mode is NavigationMode.TRANSITIONING


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_transition_duration_must_be_finite_and_positive(duration: float) -> None:
    with pytest.raises(ValueError):
        GnssModeArbitrator(duration)


def test_timestamps_must_be_finite_and_nondecreasing() -> None:
    arbitrator = GnssModeArbitrator(transition_duration_s=2.0)
    arbitrator.update(1.0, True)

    with pytest.raises(ValueError, match="nondecreasing"):
        arbitrator.update(0.9, True)
    with pytest.raises(ValueError, match="finite"):
        GnssModeArbitrator(2.0).update(float("nan"), True)
