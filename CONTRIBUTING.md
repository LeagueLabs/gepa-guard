# Contributing to gepa-guard

Thank you for your interest in gepa-guard. This guide tells you how to set up
the project, which rules a change must follow and how to send it.

For a large change, open an issue first. Agree on the approach before you
write the code.

## Set up

Install [uv](https://docs.astral.sh/uv/). Then:

```bash
git clone <your fork>
cd gepa-guard
uv sync
```

uv installs the Python version in `.python-version` and every development
dependency. Do not use pip, poetry, conda or a `requirements.txt`.

## Checks

Run these before you open a pull request. CI runs the same commands on
Python 3.11 to 3.15.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv lock --check
```

## Tests

- Write the test before the code when you add behaviour.
- Use pytest only. Do not use `unittest`, `unittest.mock` or the
  `monkeypatch` fixture.
- Keep each test at most 10 lines of code.
- Group related tests in a class named `Test<Subject>`.
- Put repeated inputs in `@pytest.mark.parametrize` and repeated setup in
  fixtures. Shared fixtures go in `conftest.py`.
- When code needs a stand-in, pass it a small compatible object. Design code
  to take its collaborators as arguments.
- Tests use no network, no model and no API key.
- Coverage must stay at or above 95 %, with branches.

```python
class TestSlugify:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("Hello World", "hello-world"), ("  a  b ", "a-b"), ("", "")],
    )
    def test_slug(self, text: str, expected: str) -> None:
        assert slugify(text) == expected
```

## Code

- Use type hints on every public function and method.
- Keep functions small, with one job each. Prefer early returns to nested
  branches.
- Prefer explicit code to implicit behaviour.
- Put configuration and shared values in a dataclass, not in module-level
  variables. Pass the instance to the code that needs it.
- Use dataclasses for new value objects.
- Write comments only when the code cannot explain itself. Choose names and
  structure that make comments unnecessary.
- ruff decides formatting. Do not format by hand.

## Docstrings

Use numpy style. State what the object does, its parameters, what it returns
and what it raises. Leave out rationale and history. Put those in the pull
request.

```python
def slugify(text: str) -> str:
    """Return a lower-case, hyphen-separated slug of `text`.

    Parameters
    ----------
    text : str
        Text to convert.

    Returns
    -------
    str
        The slug. Empty text gives an empty slug.
    """
```

## Writing style

This applies to documentation, docstrings and comments.

- Write short, plain sentences.
- Do not use em dashes or semicolons. Write two sentences.
- Do not join two statements with ", so". Write two sentences.

## Dependencies

- Add a dependency only when the standard library cannot do the job well.
- Add it with `uv add` or `uv add --dev`. Commit `uv.lock` with the change.
- Do not pin versions in `pyproject.toml`. `uv.lock` records exact versions.
  The lower bound on gepa is the one exception. It names the oldest gepa
  release this library supports.
- The project supports Python 3.11 to 3.15.

## Compatibility with gepa

- gepa is a dependency. A change to gepa-guard must not need a change to gepa.
- Use only the public gepa API.
- Only `gepa_guard._compat` imports from gepa.
- Every change must work on the oldest supported gepa and on the newest. It
  must work whether or not gepa ships
  `gepa.strategies.acceptance.NonRegressionAcceptance`.

## Public API and versions

- The public API is the names in `gepa_guard.__all__`. Other names start with
  `_`.
- The project uses [semantic versioning](https://semver.org/).
- A change that users can see needs an entry under "Unreleased" in
  `CHANGELOG.md`.

## Commits and pull requests

- Use [conventional commits](https://www.conventionalcommits.org/): `feat`,
  `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`.
- Keep one change per pull request.
- The pull request description has a short summary of the intent and a
  "Changes in this PR" list.
- Pull requests are squash merged. The squash message summarises what landed.
- Pin GitHub Actions to a full commit SHA, with the tag as a trailing comment.

## Content

- Do not add data, model outputs, prompts from real systems or credentials.
  Tests use small synthetic stand-ins.

## AI assistance

You may use AI tools to help with a contribution. Say so in the pull request.
You are responsible for every line you submit. Read it, run the checks and be
ready to explain it in review.

## Security

Do not report a vulnerability in a public issue. Use GitHub private
vulnerability reporting on this repository.
