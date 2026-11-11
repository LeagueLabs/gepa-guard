# Security Policy

## Reporting a Vulnerability

Report security vulnerabilities by opening a new [GitHub security
advisory](https://github.com/LeagueLabs/gepa-guard/security/advisories/new).

> [!WARNING]
> Do **not** file a public GitHub issue for a security report.

When you report, include as much of this as you can:
- A clear description of the issue and its potential impact.
- Steps to reproduce, or a proof of concept.
- The affected gepa-guard and gepa versions, and your environment.
- Any suggested mitigations.

If the maintainers accept the report, they prepare a patch in private and
publish a dedicated bugfix release as soon as the fix allows.

## Supported Versions

**Security fixes are applied only to the latest release.** Older releases do
not receive fixes. If you use an older version of gepa-guard, upgrade to the
latest release to receive security updates.

> [!NOTE]
> The current supported release line is 0.x (latest: 0.1.0).

When you report a potential vulnerability, confirm that it reproduces with the
latest release.

| Version            | Supported |
| ------------------ | --------- |
| Latest 0.x release | ✅        |
| Earlier releases   | ❌        |

## Scope

gepa-guard runs inside your optimization process. It calls only the
evaluation function and the base criterion that you give it. It makes no
network calls of its own. Report a vulnerability in gepa to the gepa project.
