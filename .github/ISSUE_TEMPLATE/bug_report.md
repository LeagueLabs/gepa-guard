---
name: Bug report
about: Report an error or unexpected behavior
title: "[Bug]: "
labels: ["bug"]
---

<!--
Thanks for taking the time to file a bug report.
Fill out the sections below as completely as you can.
Do not paste real data, real prompts or credentials. Use synthetic examples.
To report a vulnerability, follow SECURITY.md instead of opening an issue.
-->

## Actual behavior

<!-- Describe what happens. Include the full error output and traceback. -->

## Expected behavior

<!-- Describe what you expected to happen. -->

## Steps / code to reproduce

<!--
Give the smallest code that reproduces the issue, with synthetic data.
Replace `<your_protected_check>` with the objective name you protect.
-->

```python
from gepa.strategies.acceptance import StrictImprovementAcceptance

from gepa_guard import GuardedAcceptance

guard = GuardedAcceptance(
    StrictImprovementAcceptance(),
    protected=("<your_protected_check>",),
)
```

## Additional context

<!-- Anything else that might help, such as the gepa strategies you use. -->

## Version information

- gepa-guard: <!-- `python -c "from importlib.metadata import version; print(version('gepa-guard'))"` -->
- gepa: <!-- `python -c "from importlib.metadata import version; print(version('gepa'))"` -->
- Python: <!-- `python -V` -->
- OS: <!-- Select one: macOS / Windows / Linux -->
