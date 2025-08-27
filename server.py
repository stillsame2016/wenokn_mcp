from fastmcp import FastMCP

mcp = FastMCP("WENOKN MCP Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"
