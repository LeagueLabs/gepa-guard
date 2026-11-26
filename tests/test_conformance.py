import math
from typing import Any

import pytest
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.strategies import acceptance
from gepa.strategies.acceptance import ImprovementOrEqualAcceptance

from gepa_guard import BaseCriterion, GuardedAcceptance

PROTECTED = ("emergency",)
PASS = {"emergency": 1.0}
FAIL = {"emergency": 0.0}


def _evaluation(objectives: list[Any] | None, size: int) -> SubsampleEvaluation:
    return SubsampleEvaluation(scores=[0.0] * size, objective_scores=objectives)


def proposal(before: list[Any], after: list[Any], **fields: Any) -> CandidateProposal:
    defaults: dict[str, Any] = {
        "candidate": {"prompt": "child"},
        "parent_program_ids": [0],
        "subsample_indices": list(range(len(before))),
        "subsample_scores_before": [0.0] * len(before),
        "subsample_scores_after": [0.0] * len(after),
        "eval_before": _evaluation(before, len(before)),
        "eval_after": _evaluation(after, len(after)),
    }
    return CandidateProposal(**(defaults | fields))


def case(name: str, before: list[Any], after: list[Any], accepted: bool, **fields: Any):
    return pytest.param(before, after, fields, accepted, id=name)


CASES = [
    case("pass_kept", [PASS], [PASS], True),
    case("pass_lost", [PASS, PASS], [PASS, FAIL], False),
    case("pass_missing", [PASS], [{}], False),
    case("pass_lowered", [PASS], [{"emergency": 0.99}], False),
    case("parent_failed", [FAIL], [{}], True),
    case("unprotected_lost", [{"core": 1.0}], [{"core": 0.0}], True),
    case("unprotected_nan", [{"core": math.nan}], [{"core": "x"}], True),
    case("bool_pass", [{"emergency": True}], [PASS], True),
    case("no_parent_eval", [PASS], [PASS], False, eval_before=None),
    case("no_child_eval", [PASS], [PASS], False, eval_after=None),
    case("no_objectives", [PASS], [PASS], False, eval_after=_evaluation(None, 1)),
    case("empty", [], [], False),
    case("row_count_differs", [PASS], [PASS, PASS], False),
    case("scores_misaligned", [PASS], [PASS], False, eval_after=_evaluation([PASS], 2)),
    case("subsample_scores", [PASS], [PASS], False, subsample_scores_after=[0.0, 0.0]),
    case("indices_misaligned", [PASS], [PASS], False, subsample_indices=[0, 1]),
    case("row_not_dict", [PASS], ["row"], False),
    case("child_nan", [PASS], [{"emergency": math.nan}], False),
    case("parent_inf", [{"emergency": math.inf}], [PASS], False),
    case("string_score", [{"emergency": "1.0"}], [PASS], False),
    case("lost_then_invalid", [PASS, PASS], [FAIL, {"emergency": math.nan}], False),
]

DIFFERENT_KIND = {"lost_then_invalid", "pass_lowered", "pass_missing"}
SAME_KIND = [case for case in CASES if case.id not in DIFFERENT_KIND]

NON_BINARY = [
    pytest.param([{"emergency": 0.5}], [{}], id="parent_partial"),
    pytest.param([FAIL], [{"emergency": 0.5}], id="child_partial"),
]


@pytest.fixture
def upstream() -> Any:
    rule = getattr(acceptance, "NonRegressionAcceptance", None)
    if rule is None:
        pytest.skip("this gepa release has no NonRegressionAcceptance")
    return rule


def guard() -> GuardedAcceptance:
    return GuardedAcceptance(ImprovementOrEqualAcceptance(), PROTECTED)


class TestBaseCriterion:
    def test_gepa_criteria_match(self) -> None:
        assert isinstance(ImprovementOrEqualAcceptance(), BaseCriterion)

    def test_the_guard_is_a_criterion(self) -> None:
        assert isinstance(guard(), BaseCriterion)


class TestGuardOnGepaProposals:
    @pytest.mark.parametrize(("before", "after", "fields", "accepted"), CASES)
    def test_verdict(
        self, before: Any, after: Any, fields: Any, accepted: bool
    ) -> None:
        verdict = guard().should_accept(proposal(before, after, **fields), None)
        assert verdict is accepted


class TestConformanceWithGepa:
    @pytest.mark.parametrize(("before", "after", "fields", "accepted"), CASES)
    def test_same_verdict(
        self, upstream: Any, before: Any, after: Any, fields: Any, accepted: bool
    ) -> None:
        rule = upstream(base=ImprovementOrEqualAcceptance(), protected=PROTECTED)
        assert rule.should_accept(proposal(before, after, **fields), None) is accepted

    @pytest.mark.parametrize(("before", "after", "fields", "accepted"), SAME_KIND)
    def test_same_kind_of_rejection(
        self, upstream: Any, before: Any, after: Any, fields: Any, accepted: bool
    ) -> None:
        candidate = proposal(before, after, **fields)
        ours = guard()
        ours.should_accept(candidate, None)
        rule = upstream(base=ImprovementOrEqualAcceptance(), protected=PROTECTED)
        unverifiable = "could not be verified" in rule.reject_reason(candidate, None)
        assert bool(ours.stats.vetoed_unverifiable) is unverifiable

    @pytest.mark.parametrize("protected", ["emergency", (), ("",), ("ok", 1)])
    def test_same_invalid_protected(self, upstream: Any, protected: Any) -> None:
        with pytest.raises(ValueError, match="protected"):
            upstream(base=ImprovementOrEqualAcceptance(), protected=protected)
        with pytest.raises(ValueError, match="protected"):
            GuardedAcceptance(ImprovementOrEqualAcceptance(), protected)


class TestDifferencesFromGepa:
    @pytest.mark.parametrize(("before", "after"), NON_BINARY)
    def test_guard_rejects_non_binary_scores(self, before: Any, after: Any) -> None:
        assert guard().should_accept(proposal(before, after), None) is False

    @pytest.mark.parametrize(("before", "after"), NON_BINARY)
    def test_gepa_rule_accepts_them(
        self, upstream: Any, before: Any, after: Any
    ) -> None:
        rule = upstream(base=ImprovementOrEqualAcceptance(), protected=PROTECTED)
        assert rule.should_accept(proposal(before, after), None) is True

    @pytest.mark.parametrize("after", [{"emergency": 0.99}, {}])
    def test_pass_is_unverifiable_here_and_a_regression_in_gepa(
        self, upstream: Any, after: dict[str, float]
    ) -> None:
        changed = proposal([PASS], [after])
        ours = guard()
        ours.should_accept(changed, None)
        rule = upstream(base=ImprovementOrEqualAcceptance(), protected=PROTECTED)
        rule.should_accept(changed, None)
        assert ours.stats.vetoed_unverifiable == 1
        assert "regressed" in rule.reject_reason(changed, None)
