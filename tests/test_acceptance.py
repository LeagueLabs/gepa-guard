from typing import Any

import pytest

from gepa_guard import GuardedAcceptance, ProtectedFloor, UnguardedMergeError

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
def guard(
    accepting: Any, floor: ProtectedFloor, protected: tuple[str, ...]
) -> GuardedAcceptance:
    return GuardedAcceptance(accepting, protected, floor=floor)


class TestConstruction:
    def test_rejects_unmeasured_floor(
        self, accepting: Any, model: Any, rows: list[dict[str, str]]
    ) -> None:
        floor = ProtectedFloor(model, rows, ("emergency",))
        with pytest.raises(ValueError, match="measured"):
            GuardedAcceptance(accepting, ("emergency",), floor=floor)

    @pytest.mark.parametrize(
        "names", [("emergency",), ("encoding",), ("emergency", "encoding", "other")]
    )
    def test_rejects_floor_with_other_names(
        self, accepting: Any, floor: ProtectedFloor, names: tuple[str, ...]
    ) -> None:
        with pytest.raises(ValueError, match="the floor protects"):
            GuardedAcceptance(accepting, names, floor=floor)

    def test_accepts_floor_names_in_any_order(
        self, accepting: Any, floor: ProtectedFloor
    ) -> None:
        guard = GuardedAcceptance(accepting, ("encoding", "emergency"), floor=floor)
        assert guard.floor is floor

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

    def test_minibatch_regression_overrides_the_base(
        self, guard: GuardedAcceptance, accepting: Any, regressed_proposal: Any
    ) -> None:
        assert not guard.should_accept(regressed_proposal, STATE)
        assert (guard.stats.vetoed_subsample, accepting.calls) == (1, 1)

    @pytest.mark.parametrize(("base", "expected"), [("accepting", 1), ("rejecting", 0)])
    def test_counts_vetoes_the_base_would_accept(
        self,
        request: pytest.FixtureRequest,
        regressed_proposal: Any,
        base: str,
        expected: int,
    ) -> None:
        guard = GuardedAcceptance(request.getfixturevalue(base), ("emergency",))
        guard.should_accept(regressed_proposal, STATE)
        assert guard.stats.vetoed_subsample_base_accepted == expected

    def test_unverifiable_does_not_ask_the_base(
        self, guard: GuardedAcceptance, accepting: Any, unverifiable_proposal: Any
    ) -> None:
        guard.should_accept(unverifiable_proposal, STATE)
        assert accepting.calls == 0

    def test_counts_regressed_checks(
        self, guard: GuardedAcceptance, regressed_proposal: Any
    ) -> None:
        guard.should_accept(regressed_proposal, STATE)
        assert guard.stats.regressed["emergency"] == 1

    def test_counts_checked_passes(
        self, guard: GuardedAcceptance, regressed_proposal: Any
    ) -> None:
        guard.should_accept(regressed_proposal, STATE)
        assert guard.stats.checked["emergency"] == 2

    def test_lists_an_unmatched_name_at_zero(
        self, accepting: Any, clean_proposal: Any
    ) -> None:
        guard = GuardedAcceptance(accepting, ("emergncy",))
        guard.should_accept(clean_proposal, STATE)
        assert guard.stats.as_dict()["checked"] == {"emergncy": 0}

    def test_unverifiable_checks_nothing(
        self, guard: GuardedAcceptance, unverifiable_proposal: Any
    ) -> None:
        guard.should_accept(unverifiable_proposal, STATE)
        assert sum(guard.stats.checked.values()) == 0

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
        self,
        rejecting: Any,
        floor: ProtectedFloor,
        protected: tuple[str, ...],
        clean_proposal: Any,
    ) -> None:
        guard = GuardedAcceptance(rejecting, protected, floor=floor)
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
        assert (guard.stats.vetoed_floor, guard.stats.regressed["emergency"]) == (1, 1)
        assert guard.stats.overrode_base == 1

    @pytest.mark.parametrize("text", ["unscored-e1", "missing-e1"])
    def test_unscored_floor_case_is_unverifiable(
        self, guard: GuardedAcceptance, make_proposal: Any, text: str
    ) -> None:
        proposal = make_proposal([{}], [{}], text=text)
        assert not guard.should_accept(proposal, STATE)
        assert (guard.stats.vetoed_unverifiable, guard.stats.vetoed_floor) == (1, 0)
        expected = "could not be verified (no score for 'emergency' at floor row 0)"
        assert expected in guard.reject_reason(proposal, STATE)

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
                "protected regression in the minibatch on emergency at row 1",
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

    def test_names_the_data_id(
        self, guard: GuardedAcceptance, regressed_proposal: Any
    ) -> None:
        regressed_proposal.subsample_indices = [10, 11]
        guard.should_accept(regressed_proposal, STATE)
        expected = "on emergency at row 1 (data id 11)"
        assert expected in guard.reject_reason(regressed_proposal, STATE)

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
        guard = GuardedAcceptance(accepting, ("emergency", "encoding"), floor=floor)
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


class TestThreads:
    def test_decisions_do_not_overlap(
        self, meeting_base: Any, side_by_side: Any, make_proposal: Any
    ) -> None:
        guard = GuardedAcceptance(meeting_base, ("emergency",))
        first, second = (make_proposal([{}], [{}]) for _ in range(2))
        side_by_side(
            lambda proposal: guard.should_accept(proposal, STATE), first, second
        )
        assert not meeting_base.meeting.overlapped


class TestMerges:
    def test_merged_candidate_stops_the_run(
        self, guard: GuardedAcceptance, clean_proposal: Any, merged_state: Any
    ) -> None:
        with pytest.raises(UnguardedMergeError, match="use_merge=False"):
            guard.should_accept(clean_proposal, merged_state)

    def test_names_the_merged_candidate(
        self, guard: GuardedAcceptance, clean_proposal: Any, merged_state: Any
    ) -> None:
        with pytest.raises(UnguardedMergeError, match="candidate 3") as error:
            guard.should_accept(clean_proposal, merged_state)
        assert error.value.candidate == 3

    def test_reflective_candidates_pass(
        self, guard: GuardedAcceptance, clean_proposal: Any, reflective_state: Any
    ) -> None:
        assert guard.should_accept(clean_proposal, reflective_state)

    def test_stop_is_not_counted(
        self, guard: GuardedAcceptance, clean_proposal: Any, merged_state: Any
    ) -> None:
        with pytest.raises(UnguardedMergeError):
            guard.should_accept(clean_proposal, merged_state)
        assert guard.stats.proposals == 0


class TestReusedScores:
    @pytest.fixture
    def id_floor(
        self, model: Any, rows: list[dict[str, str]], protected: tuple[str, ...]
    ) -> ProtectedFloor:
        floor = ProtectedFloor(model, rows, protected, ids=[10, 11, 12])
        floor.measure({"prompt": "seed"})
        return floor

    @pytest.fixture
    def id_guard(
        self, accepting: Any, id_floor: ProtectedFloor, protected: tuple[str, ...]
    ) -> GuardedAcceptance:
        return GuardedAcceptance(accepting, protected, floor=id_floor)

    def test_reuses_the_minibatch_scores(
        self, id_guard: GuardedAcceptance, make_proposal: Any, model: Any
    ) -> None:
        proposal = make_proposal([{"emergency": 1.0}], [{"emergency": 1.0}])
        proposal.subsample_indices = [10]
        assert id_guard.should_accept(proposal, STATE)
        assert model.scored[3:] == ["e2", "x1"]

    def test_rejects_a_reused_regression(
        self, id_guard: GuardedAcceptance, make_proposal: Any, model: Any
    ) -> None:
        proposal = make_proposal([{"emergency": 0.0}], [{"emergency": 0.0}])
        proposal.subsample_indices = [10]
        assert not id_guard.should_accept(proposal, STATE)
        assert id_guard.stats.vetoed_floor == 1
        assert model.scored[3:] == []

    def test_scores_a_row_whose_copies_disagree(
        self, id_guard: GuardedAcceptance, make_proposal: Any, model: Any
    ) -> None:
        proposal = make_proposal(
            [{"emergency": 0.0}, {"emergency": 0.0}],
            [{"emergency": 1.0}, {"emergency": 0.0}],
        )
        proposal.subsample_indices = [10, 10]
        assert id_guard.should_accept(proposal, STATE)
        assert model.scored[3:] == ["e1", "e2", "x1"]

    @pytest.mark.parametrize("unscored", [{"emergency": None}, {}])
    def test_scores_a_row_the_minibatch_left_unscored(
        self,
        id_guard: GuardedAcceptance,
        make_proposal: Any,
        model: Any,
        unscored: dict[str, Any],
    ) -> None:
        proposal = make_proposal([{"emergency": 0.0}], [unscored])
        proposal.subsample_indices = [10]
        assert id_guard.should_accept(proposal, STATE)
        assert model.scored[3:] == ["e1", "e2", "x1"]

    def test_scores_every_row_without_indices(
        self, id_guard: GuardedAcceptance, clean_proposal: Any, model: Any
    ) -> None:
        assert id_guard.should_accept(clean_proposal, STATE)
        assert model.scored[3:] == ["e1", "e2", "x1"]
