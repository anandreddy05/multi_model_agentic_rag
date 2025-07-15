from langchain.tools import tool
import os

@tool
def save_tool(filename: str, document: str) -> str:
    """Save the document as a .txt file in the 'saved_responses' folder with the given filename."""
    print(f"[Tool: Save] Invoked ")
    if not document.strip():
        return "Document is empty. Nothing was saved."

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    folder = "saved_responses"
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, filename)

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(document.strip() + "\n")
        return f"Document saved successfully as '{file_path}'."
    except Exception as e:
        return f"Failed to save document: {str(e)}"
    