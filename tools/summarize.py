from langchain.tools import tool
@tool
def summarize_tool(document: str) -> str:
    """Summarize a given piece of text."""
    print(f"[Tool: Summarize] Invoked")
    if not document.strip():
        return "Document is empty. Nothing to summarize."
    return f"Summary: {document[:300]}..." 
