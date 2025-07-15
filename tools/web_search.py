from typing import Optional
from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
import os
from dotenv import load_dotenv

load_dotenv(override=True)

search_tool = SerpAPIWrapper(serpapi_api_key=os.getenv("SERP_API_KEY"))

@tool
def web_search(query: str) -> Optional[str]:
    """This tool is used to search results from the web using SerpAPI."""
    print(f"[Tool: web_search] Invoked with query: {query}")
    try:
        result = search_tool.run(query)
        return result
    except Exception as e:
        return f"Web search failed: {str(e)}"