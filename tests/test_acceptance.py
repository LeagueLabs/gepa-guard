from typing import Any

import pytest

from gepa_guard import GuardedAcceptance, ProtectedFloor

STATE = object()
NO_REJECTION = "no rejection was recorded for this proposal"


class SmallGuard(GuardedAcceptance):
    """Guard that keeps one unexplained rejection."""

    _max_pending = 1


@pytest.fixture
def floor(
    model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
) -> ProtectedFloor:
    floor = ProtectedFloor(model, rows, protected)
    floor.measure({"prompt": "seed"})
    return floor


@pytest.fixture
def guard(accepting: Any, floor: ProtectedFloor) -> GuardedAcceptance:
    return GuardedAcceptance(accepting, ("emergency",), floor=floor)


class TestConstruction:
    def test_rejects_unmeasured_floor(
        self, accepting: Any, model: Any, rows: list[dict[str, str]]
    ) -> None:
        floor = ProtectedFloor(model, rows, ("emergency",))
        with pytest.raises(ValueError, match="measured"):
            GuardedAcceptance(accepting, ("emergency",), floor=floor)

    def test_rejects_empty_protected(self, accepting: Any) -> None:
        with pytest.raises(ValueError, match="protected"):
            GuardedAcceptance(accepting, ())

    def test_protected_from_a_generator_still_guards(
        self, accepting: Any, regressed_proposal: Any
    ) -> None:
        guard = GuardedAcceptance(accepting, (name for name in ["emergency"]))
        assert not guard.should_accept(regressed_proposal, STATE)

    def test_keeps_protected_as_tuple(self, accepting: Any) -> None:
        assert GuardedAcceptance(accepting, ["emergency"]).protected == ("emergency",)

    def test_floor_is_optional(self, accepting: Any, clean_proposal: Any) -> None:
        assert GuardedAcceptance(accepting, ("emergency",)).should_accept(
            clean_proposal, STATE
        )


class TestShouldAccept:
    def test_accepts_clean_proposal(
        self, guard: GuardedAcceptance, clean_proposal: Any
    ) -> None:
        assert guard.should_accept(clean_proposal, STATE)
        assert guard.stats.accepted == 1

    def test_minibatch_regression_skips_base(
        self, guard: GuardedAcceptance, accepting: Any, regressed_proposal: Any
    ) -> None:
        assert not guard.should_accept(regressed_proposal, STATE)
        assert (guard.stats.vetoed_subsample, accepting.calls) == (1, 0)

    def test_counts_regressed_checks(
        self, guard: GuardedAcceptance, regressed_proposal: Any
    ) -> None:
        guard.should_accept(regressed_proposal, STATE)
        assert guard.stats.checks["emergency"] == 1

    def test_unverifiable_is_rejected(
        self, guard: GuardedAcceptance, unverifiable_proposal: Any
    ) -> None:
        assert not guard.should_accept(unverifiable_proposal, STATE)
        assert guard.stats.vetoed_unverifiable == 1

    def test_missing_candidate_with_floor_is_unverifiable(
        self, guard: GuardedAcceptance, clean_proposal: Any
    ) -> None:
        clean_proposal.candidate = {}
        assert not guard.should_accept(clean_proposal, STATE)
        assert guard.stats.vetoed_unverifiable == 1

    def test_base_rejection_skips_floor(
        self, rejecting: Any, floor: ProtectedFloor, clean_proposal: Any
    ) -> None:
        guard = GuardedAcceptance(rejecting, ("emergency",), floor=floor)
        before = floor.evaluations
        assert not guard.should_accept(clean_proposal, STATE)
        assert (guard.stats.vetoed_by_base, floor.evaluations) == (1, before)

    def test_floor_regression_is_rejected(
        self, guard: GuardedAcceptance, make_proposal: Any
    ) -> None:
        proposal = make_proposal(
            [{"emergency": 1.0}], [{"emergency": 1.0}], text="breaks-e2"
        )
        assert not guard.should_accept(proposal, STATE)
        assert (guard.stats.vetoed_floor, guard.stats.checks["emergency"]) == (1, 1)

    def test_counts_every_proposal(
        self, guard: GuardedAcceptance, clean_proposal: Any, regressed_proposal: Any
    ) -> None:
        for proposal in (clean_proposal, regressed_proposal, clean_proposal):
            guard.should_accept(proposal, STATE)
        assert (guard.stats.proposals, guard.stats.accepted) == (3, 2)


class TestRejectReason:
    @pytest.mark.parametrize(
        ("fixture", "expected"),
        [
            (
                "regressed_proposal",
                "protected regression in the minibatch on emergency at index 1",
            ),
            (
                "unverifiable_proposal",
                "could not be verified (missing subsample evaluation)",
            ),
        ],
    )
    def test_names_the_cause(
        self,
        guard: GuardedAcceptance,
        request: pytest.FixtureRequest,
        fixture: str,
        expected: str,
    ) -> None:
        proposal = request.getfixturevalue(fixture)
        guard.should_accept(proposal, STATE)
        assert expected in guard.reject_reason(proposal, STATE)

    def test_names_the_floor_row(
        self, guard: GuardedAcceptance, make_proposal: Any
    ) -> None:
        proposal = make_proposal([{}], [{}], text="breaks-e2")
        guard.should_accept(proposal, STATE)
        assert "regression in the floor on emergency at row 1" in guard.reject_reason(
            proposal, STATE
        )

    def test_does_not_call_base_again(
        self, rejecting: Any, clean_proposal: Any
    ) -> None:
        guard = GuardedAcceptance(rejecting, ("emergency",))
        guard.should_accept(clean_proposal, STATE)
        guard.reject_reason(clean_proposal, STATE)
        assert rejecting.calls == 1

    def test_uses_base_explanation(self, explaining: Any, clean_proposal: Any) -> None:
        guard = GuardedAcceptance(explaining, ("emergency",))
        guard.should_accept(clean_proposal, STATE)
        assert guard.reject_reason(clean_proposal, STATE) == "base says no"

    def test_names_base_without_explanation(
        self, rejecting: Any, clean_proposal: Any
    ) -> None:
        guard = GuardedAcceptance(rejecting, ("emergency",))
        guard.should_accept(clean_proposal, STATE)
        expected = "rejected: the base criterion (FixedBase) did not accept"
        assert guard.reject_reason(clean_proposal, STATE) == expected

    def test_unknown_proposal(
        self, guard: GuardedAcceptance, clean_proposal: Any
    ) -> None:
        assert guard.reject_reason(clean_proposal, STATE) == NO_REJECTION

    def test_accepted_proposal_has_no_reason(
        self, guard: GuardedAcceptance, clean_proposal: Any
    ) -> None:
        guard.should_accept(clean_proposal, STATE)
        assert guard.reject_reason(clean_proposal, STATE) == NO_REJECTION

    def test_explains_every_proposal_of_a_batch(
        self, guard: GuardedAcceptance, make_proposal: Any
    ) -> None:
        afters = ({}, {"emergency": 1.0}, {})
        batch = [make_proposal([{"emergency": 1.0}], [after]) for after in afters]
        verdicts = [guard.should_accept(proposal, STATE) for proposal in batch]
        reasons = [guard.reject_reason(proposal, STATE) for proposal in batch]
        assert verdicts == [False, True, False]
        assert [reason.startswith("rejected: ") for reason in reasons] == [
            True,
            False,
            True,
        ]

    def test_matches_proposals_by_identity(
        self, rejecting: Any, make_proposal: Any
    ) -> None:
        guard = GuardedAcceptance(rejecting, ("emergency",))
        guard.should_accept(make_proposal([{}], [{}]), STATE)
        assert guard.reject_reason(make_proposal([{}], [{}]), STATE) == NO_REJECTION

    def test_gives_each_reason_once(self, rejecting: Any, clean_proposal: Any) -> None:
        guard = GuardedAcceptance(rejecting, ("emergency",))
        guard.should_accept(clean_proposal, STATE)
        guard.reject_reason(clean_proposal, STATE)
        assert guard.reject_reason(clean_proposal, STATE) == NO_REJECTION

    def test_forgets_oldest_unexplained_rejection(
        self, rejecting: Any, make_proposal: Any
    ) -> None:
        guard = SmallGuard(rejecting, ("emergency",))
        first, second = make_proposal([{}], [{}]), make_proposal([{}], [{}])
        for proposal in (first, first, second):
            guard.should_accept(proposal, STATE)
        assert guard.reject_reason(first, STATE) == NO_REJECTION
        assert guard.reject_reason(second, STATE).startswith("rejected: ")

    def test_names_every_check_of_the_floor_row(
        self, accepting: Any, two_check_model: Any, make_proposal: Any
    ) -> None:
        floor = ProtectedFloor(
            two_check_model, [{"id": "m1"}], ("emergency", "encoding")
        )
        floor.measure({"prompt": "seed"})
        guard = GuardedAcceptance(accepting, ("emergency",), floor=floor)
        proposal = make_proposal([{}], [{}], text="breaks-both")
        guard.should_accept(proposal, STATE)
        expected = "on emergency at row 0, encoding at row 0"
        assert expected in guard.reject_reason(proposal, STATE)

    def test_acceptance_clears_an_earlier_rejection(
        self, rejecting: Any, clean_proposal: Any
    ) -> None:
        guard = GuardedAcceptance(rejecting, ("emergency",))
        guard.should_accept(clean_proposal, STATE)
        rejecting.accept = True
        guard.should_accept(clean_proposal, STATE)
        assert guard.reject_reason(clean_proposal, STATE) == NO_REJECTION
