import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class Evaluation:
    """Stand-in for gepa's SubsampleEvaluation."""

    scores: list[float]
    objective_scores: list[Any] | None


@dataclass
class Proposal:
    """Stand-in for gepa's CandidateProposal."""

    candidate: dict[str, str]
    eval_before: Evaluation | None
    eval_after: Evaluation | None
    subsample_scores_before: list[float] | None = None
    subsample_scores_after: list[float] | None = None
    subsample_indices: list[int] | None = None


@dataclass
class Lineage:
    """Stand-in for the parent record of gepa's GEPAState."""

    parent_program_for_candidate: list[list[int | None]]


class FixedBase:
    """Base criterion with a fixed answer that counts its calls."""

    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.calls = 0

    def should_accept(self, proposal: Any, state: Any) -> bool:
        self.calls += 1
        return self.accept


class ExplainingBase(FixedBase):
    """Base criterion that also explains its rejections."""

    def reject_reason(self, proposal: Any, state: Any) -> str:
        return "base says no"


class MinimalProposal:
    """Proposal with evaluations and none of the optional fields."""

    def __init__(self, before: Evaluation, after: Evaluation) -> None:
        self.eval_before = before
        self.eval_after = after


class TableModel:
    """Scores rows from a table of candidate text to row id to objectives."""

    def __init__(self, table: Mapping[str, Mapping[str, dict[str, float]]]) -> None:
        self.table = table
        self.scored: list[str] = []

    def __call__(
        self, rows: Sequence[Any], candidate: Mapping[str, str]
    ) -> list[dict[str, float]]:
        self.scored.extend(row["id"] for row in rows)
        return [self.table[candidate["prompt"]][row["id"]] for row in rows]


class Meeting:
    """Records whether two threads were inside a call at the same time."""

    def __init__(self) -> None:
        self.overlapped = False
        self._barrier = threading.Barrier(2, timeout=0.2)

    def join(self) -> None:
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            return
        self.overlapped = True


class MeetingBase(FixedBase):
    """Accepting base criterion that joins a meeting on each call."""

    def __init__(self) -> None:
        super().__init__(accept=True)
        self.meeting = Meeting()

    def should_accept(self, proposal: Any, state: Any) -> bool:
        self.meeting.join()
        return super().should_accept(proposal, state)


class MeetingModel(TableModel):
    """Table model that joins a meeting on each call once one is set."""

    meeting: Meeting | None = None

    def start_meeting(self) -> None:
        self.meeting = Meeting()

    def __call__(
        self, rows: Sequence[Any], candidate: Mapping[str, str]
    ) -> list[dict[str, float]]:
        if self.meeting is not None:
            self.meeting.join()
        return super().__call__(rows, candidate)


def _evaluation(objectives: list[Any]) -> Evaluation:
    return Evaluation(scores=[0.0] * len(objectives), objective_scores=objectives)


@pytest.fixture
def make_proposal() -> Callable[..., Proposal]:
    def build(before: list[Any], after: list[Any], text: str = "holds") -> Proposal:
        return Proposal({"prompt": text}, _evaluation(before), _evaluation(after))

    return build


@pytest.fixture
def clean_proposal(make_proposal: Callable[..., Proposal]) -> Proposal:
    return make_proposal([{"emergency": 1.0}], [{"emergency": 1.0}])


@pytest.fixture
def regressed_proposal(make_proposal: Callable[..., Proposal]) -> Proposal:
    return make_proposal(
        [{"emergency": 1.0}, {"emergency": 1.0}],
        [{"emergency": 1.0}, {"emergency": 0.0}],
    )


@pytest.fixture
def minimal_proposal() -> MinimalProposal:
    passed = [{"emergency": 1.0}]
    return MinimalProposal(_evaluation(passed), _evaluation(passed))


@pytest.fixture
def unverifiable_proposal(clean_proposal: Proposal) -> Proposal:
    clean_proposal.eval_before = None
    return clean_proposal


@pytest.fixture
def side_by_side() -> Callable[..., None]:
    def run(call: Callable[[Any], Any], first: Any, second: Any) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(call, [first, second]))

    return run


@pytest.fixture
def meeting_base() -> MeetingBase:
    return MeetingBase()


@pytest.fixture
def meeting_model(table: dict[str, dict[str, dict[str, float]]]) -> MeetingModel:
    return MeetingModel(table)


@pytest.fixture
def reflective_state() -> Lineage:
    return Lineage([[None], [0], [1]])


@pytest.fixture
def merged_state() -> Lineage:
    return Lineage([[None], [0], [0], [1, 2]])


@pytest.fixture
def accepting() -> FixedBase:
    return FixedBase(accept=True)


@pytest.fixture
def rejecting() -> FixedBase:
    return FixedBase(accept=False)


@pytest.fixture
def explaining() -> ExplainingBase:
    return ExplainingBase(accept=False)


@pytest.fixture
def table() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "seed": {
            "e1": {"emergency": 1.0},
            "e2": {"emergency": 1.0},
            "x1": {"encoding": 1.0},
        },
        "holds": {
            "e1": {"emergency": 1.0},
            "e2": {"emergency": 1.0},
            "x1": {"encoding": 1.0},
        },
        "breaks-e2": {
            "e1": {"emergency": 1.0},
            "e2": {"emergency": 0.0},
            "x1": {"encoding": 1.0},
        },
        "breaks-x1": {"e1": {"emergency": 1.0}, "e2": {"emergency": 1.0}, "x1": {}},
        "breaks-all": {
            "e1": {"emergency": 0.0},
            "e2": {"emergency": 0.0},
            "x1": {"encoding": 0.0},
        },
        "fails-e2": {
            "e1": {"emergency": 1.0},
            "e2": {"emergency": 0.0},
            "x1": {"encoding": 1.0},
        },
    }


@pytest.fixture
def model(table: dict[str, dict[str, dict[str, float]]]) -> TableModel:
    return TableModel(table)


@pytest.fixture
def two_check_model() -> TableModel:
    return TableModel(
        {
            "seed": {"m1": {"emergency": 1.0, "encoding": 1.0}},
            "breaks-both": {"m1": {}},
        }
    )


@pytest.fixture
def rows() -> list[dict[str, str]]:
    return [{"id": "e1"}, {"id": "e2"}, {"id": "x1"}]


@pytest.fixture
def protected() -> tuple[str, ...]:
    return ("emergency", "encoding")
