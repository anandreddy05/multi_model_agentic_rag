from sympy import sympify
from langchain.tools import tool
@tool
def calculator(query: str) -> str:
    """Calculate mathematical expressions. Use this for ANY math calculation."""
    print(f"[Tool: Calculator] Invoked with query: {query}")
    try:
        return str(sympify(query).evalf())
    except Exception as e:
        return f"Error: {e}"


    
