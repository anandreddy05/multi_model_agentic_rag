from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

@tool
def Python_Code_Executor(prompt: str) -> str:
    """This tool executes safe Python expressions. Blocks use of dangerous code like import, os, open."""
    print(f"[Tool: Python_Code_Executor] Invoked with query: {prompt}")
    blocked_keywords = ["import", "os", "open(", "__", "eval", "exec", "system"]
    
    if any(keyword in prompt for keyword in blocked_keywords):
        return "🙅‍♂️ Disallowed code: unsafe operations detected."
    if not prompt:
        return "No Prompt"
    if not prompt.startswith("print"):
        prompt = f"print({prompt})"
    
    try:
        result = python_repl.run(prompt)
        return result.strip() if result else "No Output"
    except SystemError as e:
        return f"SystemError"
    except Exception as e:
        return f"🙅‍♂️ Error during execution: {e}"
