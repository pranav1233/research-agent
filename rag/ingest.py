"""
Phase 3a: RAG Ingest Pipeline
==============================
Run this script ONCE to build your knowledge base.

It does 4 things in order:
  1. FETCH   — download paper abstracts from arXiv
  2. CHUNK   — split long text into overlapping passages
  3. EMBED   — convert each passage into a vector using a local model
  4. SAVE    — store vectors in a FAISS index on disk

After this runs, you have two files on disk:
  data/faiss.index   — the vector index (fast similarity search)
  data/chunks.json   — the raw text of every chunk (to retrieve later)

Why store both?
  FAISS only stores vectors, not the original text.
  So when FAISS says "chunk #42 is the closest match",
  we look up chunk #42 in chunks.json to get the actual words.
"""

import os
import json
import time
import arxiv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "data"                       # where to save the index + chunks
FAISS_INDEX_PATH = f"{DATA_DIR}/faiss.index"
CHUNKS_PATH = f"{DATA_DIR}/chunks.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs locally on CPU/MPS
                                         # produces 384-dimensional vectors

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
                         # overlap prevents losing context at chunk boundaries

# arXiv search queries — these define your knowledge base
# Add or change these to cover different ML topics
ARXIV_QUERIES = [
    "attention mechanism transformer natural language processing",
    "large language model fine-tuning RLHF instruction following",
    "retrieval augmented generation RAG knowledge grounding",
    "low rank adaptation LoRA parameter efficient fine-tuning",
    "chain of thought reasoning prompting language models",
]
PAPERS_PER_QUERY = 5    # 5 queries × 5 papers = 25 papers total


# ---------------------------------------------------------------------------
# Step 1: FETCH — download papers from arXiv
# ---------------------------------------------------------------------------

def fetch_papers(queries: list[str], papers_per_query: int) -> list[dict]:
    """
    Fetch paper metadata + abstracts from arXiv.

    Returns a list of dicts, each with:
      - title, authors, abstract, url, paper_id

    Note: We use abstracts only (not full PDFs) to keep things simple.
    For a deeper knowledge base, you'd download and parse the PDFs too.
    """
    client = arxiv.Client()
    papers = []
    seen_ids = set()    # deduplicate papers that appear in multiple queries

    for query in queries:
        print(f"\n  Searching arXiv: '{query}'")
        search = arxiv.Search(
            query=query,
            max_results=papers_per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for paper in client.results(search):
            if paper.entry_id in seen_ids:
                continue
            seen_ids.add(paper.entry_id)

            papers.append({
                "paper_id": paper.entry_id,
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary.replace("\n", " "),
                "url": paper.pdf_url,
            })
            print(f"    ✓ {paper.title[:70]}...")

        time.sleep(1)   # be polite to arXiv's API

    print(f"\n  Total papers fetched: {len(papers)}")
    return papers


# ---------------------------------------------------------------------------
# Step 2: CHUNK — split text into overlapping passages
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long string into overlapping chunks.

    Example with chunk_size=20, overlap=5:
      "AAAAABBBBBCCCCCDDDDDEEEEE"
       chunk 1: "AAAAABBBBBCCCCC"       (0  to 15)
       chunk 2:      "BBBBBCCCCCDDDDD"  (10 to 25)  ← 5 chars overlap
       chunk 3:           "CCCCCDDDDDEE" (20 to 35)

    The overlap ensures that if a key sentence falls at a chunk boundary,
    it appears fully in at least one chunk.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_chunks(papers: list[dict]) -> list[dict]:
    """
    Turn each paper into a set of chunks.

    Each chunk is stored as a dict with the text AND metadata
    (title, url) so we know which paper it came from later.
    """
    all_chunks = []

    for paper in papers:
        # Combine title + abstract for richer context per chunk
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "text": chunk,
                "paper_title": paper["title"],
                "paper_url": paper["url"],
                "chunk_index": i,          # which chunk within this paper
            })

    print(f"  Total chunks created: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# Step 3: EMBED — convert each chunk to a vector
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Run each chunk's text through the embedding model.

    The embedding model maps text → a fixed-size vector of floats.
    Texts with similar meaning end up with similar vectors.
    This is what makes semantic search work.

    Returns a 2D numpy array of shape (num_chunks, embedding_dim)
    e.g. for all-MiniLM-L6-v2: (num_chunks, 384)
    """
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}")
    print("  (First run downloads ~90MB — cached after that)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")

    # encode() returns a numpy array — batch processed automatically
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"  Embedding shape: {embeddings.shape}")   # (n_chunks, 384)
    return embeddings


# ---------------------------------------------------------------------------
# Step 4: SAVE — build FAISS index and save everything to disk
# ---------------------------------------------------------------------------

def save_index(chunks: list[dict], embeddings: np.ndarray) -> None:
    """
    Build a FAISS index from the embeddings and save to disk.

    FAISS (Facebook AI Similarity Search) does one thing extremely fast:
    given a query vector, find the k most similar vectors in the index.

    IndexFlatL2 = exact search using L2 (Euclidean) distance.
    Simple, accurate, fast enough for thousands of chunks.
    For millions of chunks you'd switch to an approximate index.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    dim = embeddings.shape[1]           # 384 for MiniLM
    index = faiss.IndexFlatL2(dim)

    # FAISS requires float32
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"\n  FAISS index saved → {FAISS_INDEX_PATH}")
    print(f"  Vectors in index  : {index.ntotal}")

    # Save the raw chunks separately (FAISS only stores vectors, not text)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"  Chunks saved      → {CHUNKS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_index():
    """Run the full ingest pipeline."""
    print("=" * 60)
    print("  RAG Ingest Pipeline")
    print("=" * 60)

    print("\n[1/4] Fetching papers from arXiv...")
    papers = fetch_papers(ARXIV_QUERIES, PAPERS_PER_QUERY)

    print("\n[2/4] Chunking paper text...")
    chunks = build_chunks(papers)

    print("\n[3/4] Embedding chunks...")
    embeddings = embed_chunks(chunks)

    print("\n[4/4] Saving FAISS index...")
    save_index(chunks, embeddings)

    print("\n  Done! Knowledge base is ready.")
    print(f"  Run retriever.py next to test search.")


if __name__ == "__main__":
    build_index()
