from langchain_community.document_loaders import PyPDFLoader,TextLoader,UnstructuredWordDocumentLoader
import os

def load_docs(folder_path:str):
    all_docs = []
    
    for i,file in enumerate(os.listdir(folder_path)):
        path = os.path.join(folder_path,file)
        
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(path,encoding="utf-8")
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            print(f"Document {i+1}: {path} can't be loaded as it is not in .pdf or .txt or .md")
            continue
        docs = loader.load()
        for d in docs:
            d.metadata['source'] = file
        all_docs.extend(docs)
    return all_docs