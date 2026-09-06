# Compatibility

Videopython follows semantic versioning from version 1.0. Before 1.0, a minor release
can contain incompatible changes. These changes are listed in the release notes.

## Public contracts

The supported public surface consists of:

- Python names, methods, arguments, and fields described in this reference section;
- the `VideoEdit` JSON format, operation identifiers and fields, and generated JSON
  Schema;
- `PlanErrorCode` values and the fields on `PlanError` and `PlanRepair`;
- the MCP tool names, input arguments, response shapes, and edit-plan resource described
  in [MCP server](mcp.md).

Use the import paths shown in the reference. Underscore-prefixed names, undocumented
modules, and undocumented attributes are implementation details.

## Version changes

After 1.0:

- a patch release fixes behavior without changing a documented contract;
- a minor release can add APIs, optional fields, operations, or error codes;
- a major release can remove or rename APIs, add required inputs, reject previously valid
  plans, or change the meaning of an existing field or code.

A supported Python version, operating system, or FFmpeg version is not removed in a
patch release. Such environment changes are stated in the release notes.

## Not guaranteed

Rendered media is not byte-for-byte stable across FFmpeg versions, hardware, or model
weights. Model defaults, generated content, performance, log text, and diagnostic error
messages can change without a major release. Public configuration fields, result shapes,
structured error codes, and deterministic editing behavior remain covered by the normal
version rules.
