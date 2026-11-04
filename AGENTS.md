# Instructions for coding agents

Follow [CONTRIBUTING.md](CONTRIBUTING.md). It holds every convention for this
repository. The rules below are the ones an agent must not miss.

## Before a change is complete

All of these must pass:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv lock --check
```

## Hard rules

- Write the test first. pytest only, with no mocks and no `monkeypatch`.
- Do not change gepa. Library code in `src/gepa_guard` does not import gepa.
  Tests may, and use only its public API.
- The public API is `gepa_guard.__all__`. Do not change it without an entry
  in `CHANGELOG.md`.
- Do not add data, model outputs, credentials, internal company names, URLs
  or ticket keys.
- Do not edit `uv.lock` by hand. Use `uv add`, `uv remove` or `uv lock`.
- Do not use em dashes, semicolons or a "..., so ..." clause in documentation,
  docstrings or comments. Write two sentences.
- Follow the current PEP standards. If you are unsure, check the
  [PEP index](https://peps.python.org/).
