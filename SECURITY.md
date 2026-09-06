# Security Policy

## Supported versions

Videopython is pre-1.0. Security fixes are released for the latest version only. Update
to the latest PyPI release before reporting a problem that might already be fixed.

## Report a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's
[private vulnerability report](https://github.com/BartWojtowicz/videopython/security/advisories/new)
so the report and follow-up discussion stay private.

Include:

- the affected videopython version and operating system;
- the security impact and who can trigger it;
- the smallest reproducible example;
- any known workaround or mitigation.

Do not attach private media, access tokens, or other secrets. Use synthetic media when
a file is required to reproduce the problem.

The maintainer will assess the report and coordinate any fix and disclosure through the
private advisory. There is no guaranteed response time.

## Security boundary

The MCP server is a trusted local process, not a sandbox. Its documented file and
process access is described in [MCP security boundary](docs/explanation/mcp-security.md).
Behavior inside that documented boundary is not a vulnerability by itself. Report cases
that cross the boundary, such as access beyond the process permissions, unexpected code
execution, or unsafe handling of untrusted media or plans.
