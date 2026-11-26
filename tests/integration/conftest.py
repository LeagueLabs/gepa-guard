from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gepa
import pytest
from gepa.core.adapter import EvaluationBatch

SCRIPT = ("alpha beta", "emergency alpha", "emergency alpha beta")
MERGE_SEED = {"a": "emergency gamma", "b": "none"}
MERGE_SCRIPT = {"a": "emergency alpha beta", "b": "delta"}


@dataclass(frozen=True)
class Row:
    """A task row. It passes when the prompt contains its keyword."""

    keyword: str
    protected: bool = False


EMERGENCY = Row("emergency", protected=True)
ALPHA = Row("alpha")
BETA = Row("beta")
GAMMA = Row("gamma")
DELTA = Row("delta")
UNGRADED = Row("ungraded", protected=True)
SEED = {"prompt": "emergency"}


class KeywordAdapter:
    """gepa adapter that scores a row 1.0 when the candidate holds its keyword."""

    propose_new_texts = None

    def evaluate(
        self, batch: list[Row], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        scores = [float(row.keyword in _text(candidate)) for row in batch]
        traces = [{"keyword": row.keyword} for row in batch] if capture_traces else None
        return EvaluationBatch(
            outputs=[None] * len(batch),
            scores=scores,
            trajectories=traces,
            objective_scores=self.objectives(batch, candidate),
        )

    def objectives(
        self, batch: Sequence[Row], candidate: Mapping[str, str]
    ) -> list[dict[str, float]]:
        return [
            {"emergency": float(row.keyword in _text(candidate))}
            if row.protected
            else {}
            for row in batch
        ]

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, float]]]:
        rows = [{"score": score} for score in eval_batch.scores]
        return {name: rows for name in components_to_update}


class UngradedAdapter(KeywordAdapter):
    """Keyword adapter that leaves out the protected scores it cannot grade.

    It never grades `UNGRADED`. With `children`, it grades the protected rows
    of the seed only.
    """

    def __init__(self, children: bool = False) -> None:
        self.children = children

    def objectives(
        self, batch: Sequence[Row], candidate: Mapping[str, str]
    ) -> list[dict[str, float]]:
        graded = not self.children or candidate == SEED
        return [
            scores if graded and row != UNGRADED else {}
            for row, scores in zip(
                batch, super().objectives(batch, candidate), strict=True
            )
        ]


def _text(candidate: Mapping[str, str]) -> str:
    """Join the texts of every component of a candidate."""
    return " ".join(candidate.values())


class ComponentProposer:
    """Gives the seed the texts of `MERGE_SCRIPT`. Other parents keep their texts."""

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        script = MERGE_SCRIPT if candidate == MERGE_SEED else candidate
        return {name: script[name] for name in components_to_update}


class ScriptedProposer:
    """Proposes the prompts of `SCRIPT` in order, then repeats the last one."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        prompt = SCRIPT[min(self.calls, len(SCRIPT) - 1)]
        self.calls += 1
        return {"prompt": prompt}

    def done(self, gepa_state: Any) -> bool:
        return self.calls >= len(SCRIPT)


class RecordingLogger:
    """gepa logger that keeps every line."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(message)


@pytest.fixture
def all_rows() -> list[Row]:
    return [EMERGENCY, ALPHA, BETA]


@pytest.fixture
def unprotected_rows() -> list[Row]:
    return [ALPHA, BETA]


@pytest.fixture
def protected_rows() -> list[Row]:
    return [EMERGENCY]


@pytest.fixture
def adapter() -> KeywordAdapter:
    return KeywordAdapter()


@pytest.fixture
def ungraded_valset() -> list[Row]:
    return [EMERGENCY, UNGRADED, ALPHA, BETA]


@pytest.fixture
def ungraded_adapter() -> UngradedAdapter:
    return UngradedAdapter()


@pytest.fixture
def ungraded_children_adapter() -> UngradedAdapter:
    return UngradedAdapter(children=True)


@pytest.fixture
def logger() -> RecordingLogger:
    return RecordingLogger()


@pytest.fixture
def optimize(adapter: KeywordAdapter, logger: RecordingLogger) -> Callable[..., Any]:
    def run(
        criterion: Any,
        trainset: list[Row],
        valset: list[Row] | None = None,
        scorer: KeywordAdapter | None = None,
        **options: Any,
    ) -> Any:
        proposer = ScriptedProposer()
        return gepa.optimize(
            seed_candidate=SEED,
            trainset=trainset,
            valset=valset if valset is not None else [EMERGENCY, ALPHA, BETA],
            adapter=scorer if scorer is not None else adapter,
            custom_candidate_proposer=proposer,
            reflection_minibatch_size=len(trainset),
            acceptance_criterion=criterion,
            stop_callbacks=[proposer.done],
            logger=logger,
            run_dir=None,
            track_best_outputs=False,
            seed=0,
            **options,
        )

    return run


@pytest.fixture
def optimize_with_merges(
    adapter: KeywordAdapter, logger: RecordingLogger
) -> Callable[..., Any]:
    def run(criterion: Any) -> Any:
        rows = [EMERGENCY, ALPHA, BETA, GAMMA, DELTA]
        return gepa.optimize(
            seed_candidate=MERGE_SEED,
            trainset=rows,
            valset=rows,
            adapter=adapter,
            custom_candidate_proposer=ComponentProposer(),
            reflection_minibatch_size=len(rows),
            acceptance_criterion=criterion,
            max_metric_calls=200,
            use_merge=True,
            logger=logger,
            run_dir=None,
            track_best_outputs=False,
            seed=0,
        )

    return run
