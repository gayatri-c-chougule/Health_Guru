from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
import os
import re
from dotenv import load_dotenv
from collections import Counter

load_dotenv()  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pdf_path = r"D:\112_Health_Guru\100_HG_Data\Book.pdf"

def main():
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    print(len(docs))

    all_cids = []

    for doc in docs:
        text = doc.page_content
        cids = find_cid_placeholders(text)
        if cids:
            all_cids.extend(cids)

    all_cid_counts = Counter(all_cids)

    print("\nSummary of all cid placeholders found across pages:")
    for cid, count in all_cid_counts.items():
        print(f"{cid}: {count} occurrences")

    cid_to_char = {
    "(cid:640)": "ffl",
    "(cid:637)": "ffi",
    "(cid:635)": "fl",
    "(cid:643)": "ff",
    "(cid:633)": "fi"
    }

    #todo: check if the cids are replaced properly
    for doc in docs:
        text = doc.page_content
        for cid, replacement_char in cid_to_char.items():
            text = text.replace(cid, replacement_char)
        doc.page_content = text  

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "."],
    )

    chunks = text_splitter.split_documents(docs)
    print(len(chunks))

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Number of vectors: {vector_store.index.ntotal}")
    vector_store.save_local(r"D:\112_Health_Guru\100_HG_Data\vector_db")
    print("Vector store saved to 'vector_db'")

def find_cid_placeholders(text):
    pattern = r"\(cid:\d+\)"
    return re.findall(pattern, text)

if __name__ == "__main__":
    main()
