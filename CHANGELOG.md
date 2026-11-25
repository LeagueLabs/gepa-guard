# gepa-guard CHANGELOG

## Unreleased

* Treated a `None` protected score as not scored, never as a pass, and added `UnscoredCheckError` (PR #3)

## 0.1.0a1

* Added `GuardedAcceptance`, which rejects a candidate that loses a protected pass of its parent
* Added `ProtectedFloor` and `FloorConfig` to check fixed protected rows in batches or through an adapter
* Added `VetoStats`, `Regression` and `FloorRegression` to record every decision
* Added the `BaseCriterion` and `Evaluator` protocols and `UnguardedMergeError`
* Required binary protected scores and rejected misspelled or mismatched protected names
* Made the guard and the floor thread-safe
* Published each release from one build, tested on TestPyPI before PyPI
