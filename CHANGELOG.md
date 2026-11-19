# gepa-guard CHANGELOG

## Unreleased

* Required binary protected scores and counted the vetoes that override the base criterion, to match the paper
* Added `FloorConfig.batch_size` and `ProtectedFloor.from_adapter` to score floor rows in batches through a gepa adapter
* Settled the public API: regressions name their `row` and gepa `data_id`, merges raise `UnguardedMergeError`, and `base` and `evaluate` are typed as `BaseCriterion` and `Evaluator`
* Rejected misspelled or mismatched protected names and counted the protected passes the guard checked
* Used a clearer changelog layout and named the CI test jobs by gepa version
* Made the guard and the floor thread-safe and stopped the run on gepa merges
* Added the license, authors and maintainers to the package metadata
* Adopted Contributor Covenant 3.0 as the code of conduct
* Released gepa-guard under the MIT license
* Added the release, upstream gepa and Scorecard workflows and the repository policies
* Added tests against gepa's non-regression rule and real `gepa.optimize` runs
* Added `GuardedAcceptance`, `ProtectedFloor`, `FloorConfig`, `VetoStats`, `Regression` and `FloorRegression`
* Set up the project with uv, ruff, ty, pytest and CI
