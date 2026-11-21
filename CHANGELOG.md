# gepa-guard CHANGELOG

## Unreleased

* Added the README with a quick start, run record and citation
* Required binary protected scores and counted vetoes that override the base criterion
* Added `FloorConfig.batch_size` and `ProtectedFloor.from_adapter` for batched floor scoring
* Settled the public API with `data_id`, `UnguardedMergeError`, `BaseCriterion` and `Evaluator`
* Rejected misspelled or mismatched protected names and counted checked passes
* Clarified the changelog layout and named CI test jobs by gepa version
* Made the guard and floor thread-safe and raised on gepa merges
* Added the license, authors and maintainers to the package metadata
* Adopted the Contributor Covenant 3.0 code of conduct
* Released gepa-guard under the MIT license
* Added release, upstream and Scorecard workflows and repository policies
* Tested against gepa's non-regression rule and real `gepa.optimize` runs
* Added `GuardedAcceptance` and `ProtectedFloor` with their config and records
* Set up the project with uv, ruff, ty, pytest and CI
