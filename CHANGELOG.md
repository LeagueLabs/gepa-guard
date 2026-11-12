# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `GuardedAcceptance`, a gepa acceptance criterion. It rejects a candidate
  that loses a protected pass in its minibatch or in a `ProtectedFloor`.
- `ProtectedFloor` and `FloorConfig`. They check protected rows outside the
  minibatch against a reference candidate.
- `VetoStats`, `Regression` and `FloorRegression`. They record what the guard
  found and decided.
- The MIT license.
