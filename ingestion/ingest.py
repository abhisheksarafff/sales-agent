import os
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_PATH = "./knowledge_base"
VECTORSTORE_PATH = "./vectorstore"


def detect_category(filename: str) -> str:
    fname = filename.lower()
    if any(x in fname for x in ["competitor", "battle"]):
        return "competitors"
    if any(x in fname for x in ["persona", "icp"]):
        return "personas"
    if any(x in fname for x in ["demo", "walkthrough"]):
        return "demo"
    if any(x in fname for x in ["pitch", "deck"]):
        return "pitch"
    if any(x in fname for x in ["product", "feature"]):
        return "product"
    return "general"


def ingest_documents():
    documents = []

    for root, _, files in os.walk(DOCS_PATH):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith(".txt"):
                    loader = TextLoader(path)
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(path)
                else:
                    continue
                documents.extend(loader.load())
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")

    if not documents:
        print("No documents found in knowledge_base/")
        return

    for doc in documents:
        fname = doc.metadata.get("source", "")
        doc.metadata["category"] = detect_category(fname)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=VECTORSTORE_PATH
    )
    print(f"Ingested {len(chunks)} chunks into ChromaDB")


if __name__ == "__main__":
    ingest_documents()