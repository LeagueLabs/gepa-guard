# gepa-guard

[![CI](https://github.com/LeagueLabs/gepa-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/LeagueLabs/gepa-guard/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![License](https://img.shields.io/github/license/LeagueLabs/gepa-guard?color=yellow)](https://github.com/LeagueLabs/gepa-guard/blob/main/LICENSE)

[![Python versions](https://img.shields.io/pypi/pyversions/gepa-guard)](https://pypi.org/project/gepa-guard/)
[![PyPI](https://img.shields.io/pypi/v/gepa-guard)](https://pypi.org/project/gepa-guard/)

gepa-guard is a non-regression acceptance criterion for [GEPA] reflective
prompt optimization. It rejects a candidate prompt that fails a protected
check that its parent or the seed passed.

GEPA improves a prompt against one scalar score. A candidate prompt with a
higher score can lose a rare behavior that must not break, such as the
escalation of a crisis message. GEPA accepts that candidate because its
minibatch score went up. gepa-guard keeps the GEPA objective and wraps its
acceptance criterion with protected checks.

## Why it matters

In health AI, one wrong answer on a rare case can outweigh a small gain on
average. A triage prompt that stops escalating a crisis message is not an
improvement, even when its score goes up. A prompt optimizer must not trade
that case for a higher average.

gepa-guard was developed at League Labs and funded by League Inc. It comes
from the component that gates prompt revisions for an internal health agent at
League. gepa-guard turns a silent regression into a rejected proposal with a
named cause and a record for audit, before a revised prompt reaches
deployment.

## Features

With gepa-guard you can:

- Reject a candidate that loses a protected pass of its parent in the minibatch.
- Check candidates against a fixed set of protected rows (the floor).
- Reject a proposal whose protected scores cannot be read, instead of guessing.
- Read the failed check and row in the gepa log for each protected rejection.
- Count every decision in a run record that matches the guarded proposal
  outcomes in the paper.
- Keep your base criterion, for example `StrictImprovementAcceptance`.

## Installation

gepa-guard is available via `pip`:

```bash
pip install gepa-guard
```

gepa-guard supports Python 3.11 to 3.15 and gepa 0.1.4 or later.

## Basic usage

Your gepa adapter must return `objective_scores` in its `EvaluationBatch`.
This is a list with one mapping per row. Give each protected check a score of
`1.0` (pass) or `0.0` (fail) on the rows that test it. Leave a check out of a
row when the row does not test it, or when the row has no score for it, for
example after an error. The guard also reads `None` as no score, but gepa
0.1.4 stops the run when a `valset` score is `None`. Leave the check out
instead.

The following example protects an `emergency` check in the minibatch.
`seed`, `trainset`, `valset`, `adapter` and `reflection_lm` are your usual
gepa inputs:

```python
import gepa
from gepa.strategies.acceptance import StrictImprovementAcceptance
from gepa_guard import GuardedAcceptance

guard = GuardedAcceptance(StrictImprovementAcceptance(), protected=["emergency"])

result = gepa.optimize(
    seed_candidate=seed,
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    reflection_lm=reflection_lm,
    acceptance_criterion=guard,
    max_metric_calls=500,
)

print(guard.stats.as_dict())
```

gepa logs the reason for each rejection, for example:

```text
Iteration 3: rejected: protected regression in the minibatch on emergency at row 1 (data id 11)
```

### Add a protected floor

The minibatch holds only a few rows. A floor checks each candidate that
passes the minibatch check and the base criterion against a fixed set of
protected rows. Measure the floor with the seed first. The seed defines which
checks each row must keep.

`protected_rows` is a list of rows in the form your adapter takes. Do not use
rows from your test set.

```python
from gepa.strategies.acceptance import StrictImprovementAcceptance
from gepa_guard import FloorConfig, GuardedAcceptance, ProtectedFloor

floor = ProtectedFloor.from_adapter(
    adapter, protected_rows, protected=["emergency"], config=FloorConfig(batch_size=8)
)
floor.measure(seed)

guard = GuardedAcceptance(
    StrictImprovementAcceptance(), protected=["emergency"], floor=floor
)
```

To score the floor without an adapter, use
`ProtectedFloor(evaluate, rows, protected=[...])`. `evaluate` takes a sequence
of rows and a candidate, and returns one objective mapping per row.

If you already have the seed scores of the floor rows, use
`floor.measure_scores(scores)` instead of `floor.measure(seed)`. It takes one
objective mapping per row and does not score the seed again. The floor then
protects the same passes that your own seed evaluation shows.

When some floor rows are also in your `trainset`, give the floor the gepa data
id of each row with `ids`. For a list `trainset`, the data id is the position
of the row in the `trainset`. Do not use positions in the `valset`. A wrong id
makes the floor use the scores of another row. Use `None` for a row that is
not in the `trainset`.
The guard then gives the floor the minibatch scores of the candidate. The floor
uses them for its rows in the minibatch and scores only the other rows. It
also scores a row when the minibatch has no score for one of its protected
checks.

```python
floor = ProtectedFloor.from_adapter(
    adapter, protected_rows, protected=["emergency"], ids=[0, 4, None]
)
```

## How it works

For each proposal, the guard runs these checks in order. The first check that
fails rejects the proposal.

1. **Read**: the minibatch objective scores of the parent and the candidate
   must exist and align row by row. Each protected score must be 0, 1 or
   `None`. If not, the proposal is *unverifiable* and is rejected.
2. **Minibatch**: the candidate must keep every protected pass of its parent.
3. **Base**: the base criterion, for example `StrictImprovementAcceptance`,
   must accept.
4. **Floor**: the candidate must keep every protected pass of the seed on the
   floor rows, when a floor is given.

A missing score never counts as a pass. When the candidate has no score for a
protected pass of its parent, or of the seed on the floor, and has no
regression, the proposal is *unverifiable* and is rejected. A check that is
left out or is `None` has no score.

After a minibatch rejection, the guard still asks the base criterion. The
answer does not change the decision. It only counts the rejections that
override GEPA.

## Requirements

- **Binary protected scores.** A protected score must be 0, 1 or `None`. Only
  1 is a pass. Any other value makes a minibatch proposal unverifiable. On the
  floor, it raises `ValueError`.
- **Deterministic scores.** A check that passes and fails at random on the
  same prompt causes false rejections. Make the task model as deterministic
  as you can, for example with temperature 0.
- **A seed that passes.** The guard protects only what the parent or the seed
  passed. `ProtectedFloor.measure` raises `ValueError` when the seed passes no
  row of a protected check.
- **No merges.** gepa accepts merged candidates without the acceptance
  criterion. Keep `use_merge=False`, the default of `gepa.optimize`. With
  `optimize_anything`, leave `GEPAConfig.merge` unset. If the gepa state holds
  a merged candidate, the guard raises `UnguardedMergeError` at its next
  decision.
- **One guard per run.** The guard keeps state for one run. Calls from several
  threads run one at a time.

## Reading the run record

`guard.stats` is a `VetoStats`. Its counts match the columns of the guarded
proposal outcomes table in the paper.

| Field | Meaning |
|---|---|
| `proposals`, `accepted` | Proposals decided and accepted |
| `vetoed_unverifiable` | Rejected because the protected scores could not be read, or had no score for a protected pass |
| `vetoed_subsample` | Rejected for a protected regression in the minibatch |
| `vetoed_by_base` | Rejected by the base criterion |
| `vetoed_floor` | Rejected for a protected regression in the floor |
| `vetoed_by_guard` | `vetoed_subsample` plus `vetoed_floor` |
| `vetoed_subsample_base_accepted` | Minibatch rejections of proposals that the base criterion accepts. These are the counts in parentheses in the paper |
| `overrode_base` | `vetoed_subsample_base_accepted` plus `vetoed_floor`. This is the "Overrode GEPA" column |
| `checked` | Protected passes of the parent compared in the minibatch, per check |
| `regressed` | Protected regressions found in the minibatch or the floor, per check |

`as_dict()` returns every count as a plain dictionary for logs and reports.

## Compatibility

- gepa 0.1.4 or later. CI tests the oldest supported and the latest release
  with `gepa.optimize`.
- `optimize_anything` also takes the guard, as
  `EngineConfig(acceptance_criterion=guard)` from `gepa.optimize_anything`.
  Return the protected scores in `side_info["scores"]`.
- Any base criterion with a `should_accept(proposal, state)` method. gepa's
  own criteria match the `BaseCriterion` protocol.

## Citation

If you use gepa-guard in your research, please cite the paper:

```bibtex
@inproceedings{barreto-ojeda2026demo,
  title     = {Demo: Blocking Silent Safety Regressions in Prompt Optimization for Health {AI}},
  author    = {Estefania Barreto-Ojeda and Aiden Arnold},
  booktitle = {The Third Workshop on GenAI for Health: Agentic Systems, Clinical Trust, and Future Potential - Demonstration Paper Track},
  year      = {2026},
  url       = {https://openreview.net/forum?id=3iBtYkwQSc},
}
```

## Contributing

Contributions are very welcome!

Please read the [Contributing] guide before you open a pull request. It
describes the development setup with [uv], the checks, the code style and the
AI assistance policy.

To set up a development environment, fork the repository, clone your fork and
run the tests:

```bash
git clone https://github.com/<your-username>/gepa-guard.git
cd gepa-guard
uv sync
uv run pytest
```

To report a vulnerability, please see the [Security] policy.

## License

Source code included in this project is available in the GitHub repository
https://github.com/LeagueLabs/gepa-guard under the MIT License (see
[LICENSE]). The project is linked to a [Code of Conduct].

[GEPA]: https://github.com/gepa-ai/gepa
[uv]: https://docs.astral.sh/uv/
[Contributing]: https://github.com/LeagueLabs/gepa-guard/blob/main/CONTRIBUTING.md
[Security]: https://github.com/LeagueLabs/gepa-guard/blob/main/SECURITY.md
[LICENSE]: https://github.com/LeagueLabs/gepa-guard/blob/main/LICENSE
[Code of Conduct]: https://github.com/LeagueLabs/gepa-guard/blob/main/CODE_OF_CONDUCT.md
