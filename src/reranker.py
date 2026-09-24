from sentence_transformers import CrossEncoder
from typing import List, Dict

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
 
# How many chunks to keep after reranking
TOP_N = 5
 
 
# Function 1: Load the reranker model 
def load_reranker() -> CrossEncoder:
    """
    Load the cross-encoder model.
    Call this once at startup — not on every request.
 
    Returns:
        CrossEncoder model ready to use
    """
    print(f"Loading reranker model: {RERANKER_MODEL}")
    model = CrossEncoder(RERANKER_MODEL)
    print("Reranker model loaded")
    return model
 
 
# Function 2: Rerank chunks 
def rerank(question: str, chunks: List[Dict], model: CrossEncoder) -> List[Dict]:
    """
    Rerank chunks by relevance to the question.
 
    How it works:
    1. Create pairs of (question, chunk_content) for each chunk
    2. Cross-encoder scores each pair — higher score = more relevant
    3. Sort chunks by score descending
    4. Return top TOP_N chunks
 
    Args:
        question: the user's question
        chunks: list of chunks from vector search (each is a dict)
        model: the loaded CrossEncoder model
 
    Returns:
        top TOP_N chunks reranked by relevance
    """
 
    if not chunks:
        return []
 
    # Create question-chunk pairs for the cross-encoder
    # The cross-encoder reads both together and scores relevance
    pairs = [(question, chunk["content"]) for chunk in chunks]
 
    # Score each pair
    # scores is a list of floats — one score per chunk
    scores = model.predict(pairs)
 
    # Attach scores to chunks
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
 
    # Sort by score descending (highest score first)
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
 
    # Return only the top N chunks
    return reranked[:TOP_N]
 
 
# Test it
if __name__ == "__main__":

 
    # Load reranker and rerank
    reranker = load_reranker()
    reranked = rerank(question, test_chunks, reranker)
 
    print(f"\nAfter reranking (best first):")
    for i, chunk in enumerate(reranked):
        print(f"  {i+1}. Section {chunk['section']} — {chunk['title']} (score: {chunk['rerank_score']:.3f})")
 