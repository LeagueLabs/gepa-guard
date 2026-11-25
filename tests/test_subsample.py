import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest

from gepa_guard import Regression
from gepa_guard._subsample import SubsampleCheck

Mutation = Callable[[Any], None]


def _set(path: str, value: Any) -> Mutation:
    def mutate(proposal: Any) -> None:
        *parents, name = path.split(".")
        target = proposal
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, name, value)

    return mutate


@pytest.fixture
def check() -> SubsampleCheck:
    return SubsampleCheck(("emergency",))


class TestConstruction:
    @pytest.mark.parametrize("protected", ["emergency", (), ("",)])
    def test_rejects_invalid_protected(self, protected: Any) -> None:
        with pytest.raises(ValueError, match="protected"):
            SubsampleCheck(protected)


class TestRegressions:
    def test_clean_when_pass_is_kept(
        self, check: SubsampleCheck, clean_proposal: Any
    ) -> None:
        assert check.inspect(clean_proposal).clean

    def test_finds_lost_pass(
        self, check: SubsampleCheck, regressed_proposal: Any
    ) -> None:
        assert check.inspect(regressed_proposal).regressions == (
            Regression(1, "emergency"),
        )

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            ({"emergency": 0.0}, {}),
            ({}, {}),
            ({"core": 1.0}, {"core": 0.0}),
        ],
    )
    def test_ignores_rows_without_a_parent_pass(
        self,
        check: SubsampleCheck,
        make_proposal: Any,
        before: dict[str, float],
        after: dict[str, float],
    ) -> None:
        assert check.inspect(make_proposal([before], [after])).clean

    @pytest.mark.parametrize("after", [{}, {"emergency": 0.0}])
    def test_missing_or_lower_score_regresses(
        self, check: SubsampleCheck, make_proposal: Any, after: dict[str, float]
    ) -> None:
        finding = check.inspect(make_proposal([{"emergency": 1.0}], [after]))
        assert finding.regressions == (Regression(0, "emergency"),)

    def test_records_the_data_id(
        self, check: SubsampleCheck, regressed_proposal: Any
    ) -> None:
        regressed_proposal.subsample_indices = [10, 11]
        assert check.inspect(regressed_proposal).regressions == (
            Regression(1, "emergency", data_id=11),
        )

    def test_counts_checked_passes(
        self, check: SubsampleCheck, regressed_proposal: Any
    ) -> None:
        assert check.inspect(regressed_proposal).checked == Counter({"emergency": 2})

    def test_rows_without_a_parent_pass_are_not_checked(
        self, check: SubsampleCheck, make_proposal: Any
    ) -> None:
        proposal = make_proposal([{"emergency": 0.0}, {}], [{}, {}])
        assert check.inspect(proposal).checked == Counter()

    def test_reports_every_check(self, make_proposal: Any) -> None:
        both = {"emergency": 1.0, "encoding": 1.0}
        finding = SubsampleCheck(("emergency", "encoding")).inspect(
            make_proposal([both], [{}])
        )
        assert [r.check for r in finding.regressions] == ["emergency", "encoding"]

    def test_unprotected_values_are_not_validated(
        self, check: SubsampleCheck, make_proposal: Any
    ) -> None:
        proposal = make_proposal([{"core": math.nan}], [{"core": "bad"}])
        assert check.inspect(proposal).clean


class TestUnverifiable:
    @pytest.mark.parametrize(
        ("mutate", "problem"),
        [
            (_set("eval_before", None), "missing subsample evaluation"),
            (_set("eval_after", None), "missing subsample evaluation"),
            (_set("eval_before.objective_scores", None), "missing objective_scores"),
            (_set("eval_after.objective_scores", []), "empty or mismatched"),
            (_set("eval_after.scores", [0.0, 0.0]), "evaluation scores"),
            (_set("subsample_scores_before", [0.0, 0.0]), "subsample scores"),
            (_set("subsample_indices", [0, 1]), "subsample indices"),
            (
                _set("eval_after.objective_scores", ["row"]),
                "invalid objectives at minibatch row 0",
            ),
            (
                _set("eval_after.objective_scores", [{"emergency": math.nan}]),
                "non-binary score nan for 'emergency' at minibatch row 0",
            ),
            (
                _set("eval_before.objective_scores", [{"emergency": "1.0"}]),
                "non-binary score '1.0' for 'emergency'",
            ),
            (
                _set("eval_before.objective_scores", [{"emergency": 0.5}]),
                "non-binary score 0.5 for 'emergency'",
            ),
            (
                _set("eval_after.objective_scores", [{"emergency": 0.99}]),
                "non-binary score 0.99 for 'emergency'",
            ),
        ],
    )
    def test_reports_problem(
        self, check: SubsampleCheck, clean_proposal: Any, mutate: Mutation, problem: str
    ) -> None:
        mutate(clean_proposal)
        assert problem in (check.inspect(clean_proposal).problem or "")

    def test_problem_hides_regressions(
        self, check: SubsampleCheck, regressed_proposal: Any
    ) -> None:
        regressed_proposal.subsample_indices = [0]
        finding = check.inspect(regressed_proposal)
        assert (finding.regressions, finding.clean) == ((), False)

    def test_problem_checks_nothing(
        self, check: SubsampleCheck, regressed_proposal: Any
    ) -> None:
        regressed_proposal.subsample_indices = [0]
        assert check.inspect(regressed_proposal).checked == Counter()

    def test_reads_object_without_evaluations(self, check: SubsampleCheck) -> None:
        assert check.inspect(object()).problem == "missing subsample evaluation"

    def test_reads_proposal_without_optional_fields(
        self, check: SubsampleCheck, minimal_proposal: Any
    ) -> None:
        assert check.inspect(minimal_proposal).clean

    @pytest.mark.parametrize("name", ["scores", "objective_scores"])
    def test_evaluation_without_a_field_is_unverifiable(
        self, check: SubsampleCheck, minimal_proposal: Any, name: str
    ) -> None:
        delattr(minimal_proposal.eval_after, name)
        assert check.inspect(minimal_proposal).problem is not None


class TestUnscored:
    def test_unscored_parent_pass_is_unverifiable(
        self, check: SubsampleCheck, make_proposal: Any
    ) -> None:
        proposal = make_proposal([{"emergency": 1.0}], [{"emergency": None}])
        expected = "no score for 'emergency' at minibatch row 0"
        assert check.inspect(proposal).problem == expected

    @pytest.mark.parametrize("before", [{}, {"emergency": 0.0}, {"emergency": None}])
    def test_unscored_without_parent_pass_is_clean(
        self, check: SubsampleCheck, make_proposal: Any, before: dict[str, Any]
    ) -> None:
        proposal = make_proposal([before], [{"emergency": None}])
        assert check.inspect(proposal).clean

    def test_unscored_parent_is_not_a_pass(
        self, check: SubsampleCheck, make_proposal: Any
    ) -> None:
        proposal = make_proposal([{"emergency": None}], [{"emergency": 0.0}])
        assert check.inspect(proposal).clean

    def test_regression_takes_precedence(
        self, check: SubsampleCheck, regressed_proposal: Any
    ) -> None:
        regressed_proposal.eval_after.objective_scores[0] = {"emergency": None}
        finding = check.inspect(regressed_proposal)
        assert (finding.problem, finding.regressions) == (
            None,
            (Regression(1, "emergency"),),
        )

    def test_unscored_pass_is_not_checked(
        self, check: SubsampleCheck, regressed_proposal: Any
    ) -> None:
        regressed_proposal.eval_after.objective_scores[0] = {"emergency": None}
        assert check.inspect(regressed_proposal).checked == Counter(emergency=1)
