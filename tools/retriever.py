from langchain.tools import tool
from rag.embed_and_index import retriever

@tool
def retriever_tool(query: str) -> str:
    """
    Retrieves relevant chunks from the knowledge base based on user query.
    """
    print(f"[Tool: Retriever] Invoked with query: {query}")
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return f"No relevant information found in the knowledge base for query: '{query}'"

    docs = "\n\n".join([
        f"📄 Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in retrieved_docs
    ])
    return docs