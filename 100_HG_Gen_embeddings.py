#region Imports
import os
import re
from dotenv import load_dotenv
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
#endregion

#region Config / Env
load_dotenv()  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Loaded from .env to avoid hard-coding secrets
pdf_path = os.getenv("PDF_PATH")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
#endregion

#region Pipeline
def main():
    """
    Process a PDF by replacing CID (Character ID) placeholders, chunking the text,
    embedding it into vectors, and storing the result in a FAISS database.

    Steps performed:
        1. Load the PDF document.
        2. Identify and count all CID placeholders across pages.
        3. Replace known CIDs with their correct characters.
        4. Split the cleaned text into overlapping chunks for embedding.
        5. Create vector embeddings using a language model.
        6. Store the vectors in a FAISS database and save locally.

    Args:
        None

    Returns:
        None
    """
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    print(len(docs))  # sanity check: number of pages loaded

    all_cids = []

    for doc in docs:
        text = doc.page_content
        cids = find_cid_placeholders(text)
        if cids:
            all_cids.extend(cids)

    all_cid_counts = Counter(all_cids)

    # sanity check 
    print("\nSummary of all cid placeholders found across pages:")
    for cid, count in all_cid_counts.items():
        print(f"{cid}: {count} occurrences")

    # dictionary for replacing the cids
    # NOTE: This CID→text map fixes common ligatures from PDF extraction (ffl/ffi/fl/ff/fi).
    # Keep this list project-specific.
    cid_to_char = {
        "(cid:640)": "ffl",
        "(cid:637)": "ffi",
        "(cid:635)": "fl",
        "(cid:643)": "ff",
        "(cid:633)": "fi"
    }

    # todo: check if the cids are replaced properly
    for doc in docs:
        text = doc.page_content
        for cid, replacement_char in cid_to_char.items():
            text = text.replace(cid, replacement_char)
        doc.page_content = text  

    # NOTE: chunk_size=1000, chunk_overlap=200 chosen to balance semantic coherence and recall.
    # Larger chunks preserve context; 200 overlap helps avoid splitting mid-topic.
    # Separators prioritize paragraph breaks and sentences to reduce mid-sentence splits.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "."],
    )

    chunks = text_splitter.split_documents(docs)
    print(len(chunks))  # sanity check: number of chunks

    # NOTE: text-embedding-3-small is a cost-effective default with good quality for retrieval.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Create and persist FAISS index locally for fast similarity search in the app.
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Number of vectors: {vector_store.index.ntotal}")
    vector_store.save_local(VECTOR_DB_PATH)
    print("Vector store saved to 'vector_db'")
#endregion

#region Utilities
def find_cid_placeholders(text):
    """
    Extract all CID (Character ID) placeholders from the given text.

    CID placeholders are typically strings in the format "(cid:###)",
    where ### is a numeric identifier for a special character in PDFs.

    Args:
        text (str): The input text to search for CID placeholders.

    Returns:
        list[str]: A list of all CID placeholder matches found in the text.
                   Returns an empty list if no placeholders are found.
    """
    pattern = r"\(cid:\d+\)"
    return re.findall(pattern, text)
#endregion

#region Entry point
if __name__ == "__main__":
    main()
#endregion
