"""videopython MCP server: the auto-edit primitives exposed as MCP tools.

Run with the ``videopython-mcp`` console script (needs the ``[ai,mcp]`` extras).
The MCP client's own model is the planner; this server only exposes the steps
(analyze -> catalog -> validate/repair/run over an EditPlan by scene id).
"""
