import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import threading

load_dotenv()

PAPERS_DIR = "./papers"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
BATCH_SIZE = 64
MAX_WORKERS = 4

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("influencer_papers")

# Thread lock for safe concurrent writes to ChromaDB
db_lock = threading.Lock()


def get_ingested_papers():
    """Returns set of filenames already in the DB."""
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return set()
    return set(m["source"] for m in results["metadatas"])


def extract_text_from_pdf(pdf_path):
    """Extract text from each page of a PDF."""
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                text_chunks.append({
                    "text": text,
                    "page": page_num
                })
    return text_chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def process_single_pdf(pdf_file):
    """Process one PDF: extract, chunk, embed, store."""
    pdf_path = os.path.join(PAPERS_DIR, pdf_file)
    paper_name = os.path.splitext(pdf_file)[0]

    print(f"  → Processing: {pdf_file}")

    try:
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            print(f"  ⚠ No extractable text found in {pdf_file} — skipping.")
            return 0

        all_chunks = []
        metadatas = []
        ids = []

        for page_data in pages:
            chunks = chunk_text(page_data["text"])
            for i, chunk in enumerate(chunks):
                chunk_id = f"{paper_name}_p{page_data['page']}_c{i}"
                all_chunks.append(chunk)
                metadatas.append({
                    "paper": paper_name,
                    "page": page_data["page"],
                    "source": pdf_file
                })
                ids.append(chunk_id)

        if not all_chunks:
            print(f"  ⚠ No chunks generated from {pdf_file} — skipping.")
            return 0

        # Embed in batches with progress bar
        embeddings = model.encode(
            all_chunks,
            batch_size=BATCH_SIZE,
            show_progress_bar=False
        ).tolist()

        # Thread-safe write to ChromaDB
        with db_lock:
            collection.add(
                documents=all_chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        print(f"  ✓ {pdf_file}: {len(all_chunks)} chunks from {len(pages)} pages")
        return len(all_chunks)

    except Exception as e:
        print(f"  ✗ Error processing {pdf_file}: {e}")
        return 0


def ingest_papers():
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in ./papers folder.")
        return

    # Check which papers are already ingested
    already_ingested = get_ingested_papers()
    new_files = [f for f in pdf_files if f not in already_ingested]

    print(f"Papers found:      {len(pdf_files)}")
    print(f"Already ingested:  {len(already_ingested)}")
    print(f"New to process:    {len(new_files)}\n")

    if not new_files:
        print("✓ All papers already ingested. Nothing to do.")
        print(f"  Total chunks in DB: {collection.count()}")
        return

    print(f"Starting ingestion with {MAX_WORKERS} workers...\n")

    total_chunks = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(process_single_pdf, new_files)
        for count in results:
            total_chunks += count

    print(f"\n✓ Done!")
    print(f"  New chunks added:     {total_chunks}")
    print(f"  Total chunks in DB:   {collection.count()}")


if __name__ == "__main__":
    ingest_papers()