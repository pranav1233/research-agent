"""
Phase 3b: RAG Retriever
========================
This is the QUERY side of RAG — used at agent runtime.

While ingest.py runs once to BUILD the index,
retriever.py runs every time the agent calls rag_search().

It does 3 things:
  1. LOAD    — read the FAISS index + chunks from disk (done once at startup)
  2. EMBED   — convert the query string into a vector
  3. SEARCH  — ask FAISS for the top-k closest chunks, return their text

How semantic search works:
  Your query "how does attention work" becomes a 384-dim vector.
  FAISS finds the chunks whose vectors are closest to that query vector.
  "Closest" means similar meaning — not just matching keywords.
  That's what makes RAG powerful over plain keyword search.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config  (must match ingest.py)
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = "data/faiss.index"
CHUNKS_PATH      = "data/chunks.json"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # same model used during ingest
TOP_K            = 4                      # how many chunks to return per query

# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """
    Loads the FAISS index once and answers semantic search queries.

    We use a class (not bare functions) so the model and index are
    loaded once at startup and reused across all queries — not
    reloaded on every tool call, which would be very slow.
    """

    def __init__(self):
        self._loaded = False
        self.index   = None
        self.chunks  = None
        self.model   = None

    def load(self):
        """
        Load the FAISS index, chunks, and embedding model from disk.
        Called once when the agent starts up.
        """
        if self._loaded:
            return  # already loaded, skip

        print("  [RAG] Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("  [RAG] Loading chunks...")
        with open(CHUNKS_PATH) as f:
            self.chunks = json.load(f)

        print(f"  [RAG] Loading embedding model ({EMBEDDING_MODEL})...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"  [RAG] Ready — {self.index.ntotal} vectors, {len(self.chunks)} chunks")
        self._loaded = True

    def search(self, query: str, top_k: int = TOP_K) -> str:
        """
        Embed the query and find the top-k most relevant chunks.

        Args:
            query:  The search string from the agent.
            top_k:  How many chunks to return.

        Returns:
            A formatted string with the most relevant passages.
            The agent reads this text to formulate its answer.
        """
        if not self._loaded:
            self.load()

        # Embed the query — same model, same vector space as the chunks
        # reshape(1, -1) because FAISS expects a 2D array (batch of queries)
        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = query_vector.astype(np.float32)

        # FAISS search — returns distances and indices of nearest neighbours
        # distances shape: (1, top_k)  — L2 distance to each result
        # indices shape:   (1, top_k)  — position of each result in the index
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx == -1:
                # FAISS returns -1 if there aren't enough vectors in the index
                continue

            chunk = self.chunks[idx]

            # Lower L2 distance = more similar
            # We convert to a 0-1 relevance score for readability
            relevance = round(1 / (1 + float(dist)), 3)

            results.append(
                f"[Result {rank}] (relevance: {relevance})\n"
                f"Source : {chunk['paper_title']}\n"
                f"URL    : {chunk['paper_url']}\n"
                f"Passage: {chunk['text']}\n"
            )

        if not results:
            return f"[NO RESULTS] No relevant passages found for: '{query}'"

        header = f"RAG search results for: '{query}'\n{'─'*50}\n"
        return header + "\n".join(results)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# One Retriever instance shared across the whole program.
# tools/rag_search.py imports this and calls retriever.search().

retriever = Retriever()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing retriever...\n")
    retriever.load()

    test_queries = [
        "how does the attention mechanism work",
        "what is LoRA fine-tuning",
        "retrieval augmented generation",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print('='*60)
        print(retriever.search(q, top_k=2))
