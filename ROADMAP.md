# Roadmap

This roadmap describes the direction of videopython. It is not a release schedule and
does not promise dates. GitHub issues track concrete work, and release notes describe
completed changes.

## Direction

Videopython aims to make deterministic, local-first video editing practical from both
Python code and AI agents. The edit plan is the central contract: a person or agent can
author structured JSON, validate it without decoding frames, repair mechanical errors,
and render it through one bounded-memory engine.

The project prioritizes:

- predictable Python and JSON interfaces;
- bounded-memory processing for long media;
- structured validation that agents can act on;
- local AI with explicit model and hardware requirements;
- small core installs, with model dependencies kept in optional extras;
- reproducible examples and measured behavior.

## Path to 1.0

Version 1.0 means that documented interfaces are stable enough for applications and
agents to depend on them. It does not mean that every possible video workflow is
implemented.

### Define the stable surface

The [compatibility policy](docs/reference/compatibility.md) defines the supported public
surface and versioning rules. Before the first release candidate, audit each documented
contract against the implementation and ensure that the reference contains only APIs
that are ready to stabilize.

The remaining pre-1.0 cleanup must resolve known validation and rendering differences.

### Strengthen AI and agent verification

Unit tests use local fakes and cannot prove that external models still work. The 1.0
release gate therefore needs evidence from the real model stack:

- run the complete MCP analyze, catalog, validate, and render flow on representative
  media;
- make the required real-model sign-off visible in the release process;

### Validate the release candidate

Publish `1.0.0rc1` before the stable release. Use it from a clean consumer environment
and through at least one external MCP agent. The release candidate period is for fixes,
documentation, and compatibility corrections, not new features or interface redesigns.

## After 1.0

Development after 1.0 will focus on a small number of themes:

- agent-native distribution, including OpenClaw and other MCP clients;
- more reliable local-model compatibility and clearer runtime diagnostics;
- richer editing operations that preserve the streaming and plan-validation contracts;
- performance improvements for long-form and high-resolution media;
- broader platform support when it can be tested continuously.

New features can remain experimental until their contracts are clear. Stable interfaces
change incompatibly only in a new major release.

## Non-goals

Videopython does not aim to become:

- a hosted inference service;
- an interactive non-linear editing application;
- a compatibility layer for every model, accelerator, or FFmpeg build;
- a second in-memory execution engine alongside the streaming engine.

The roadmap changes when project direction changes. Detailed implementation checklists
do not belong here.
