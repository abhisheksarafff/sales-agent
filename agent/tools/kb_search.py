from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool

VECTORSTORE_PATH = "./vectorstore"


@tool
def search_internal_kb(query: str) -> str:
    """Search the internal sales knowledge base for product info,
    personas, competitors, demo guides, pitch materials, and use cases.
    Always use this FIRST before any web search."""

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
    )
    results = vectorstore.similarity_search_with_score(query, k=5)

    if not results:
        return "No relevant information found in internal knowledge base."

    output = []
    for doc, score in results:
        if score > 1.5:  # Lower score = better match in this model
            continue
        category = doc.metadata.get("category", "general")
        source = doc.metadata.get("source", "unknown")
        output.append(f"[{category.upper()}] (source: {source})\n{doc.page_content}")

    if not output:
        return "LOW_CONFIDENCE: Internal KB results were not relevant enough."

    return "\n\n---\n\n".join(output)