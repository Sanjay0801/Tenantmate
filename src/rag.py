import os
import psycopg2
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
#from src.reranker import load_reranker,rerank
try:
    from src.reranker import load_reranker, rerank
except ModuleNotFoundError:
    from reranker import load_reranker, rerank
 
#Load environment variables 
load_dotenv()
 
# Config 
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}
 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
 
# Number of sections to retrieve from the database
TOP_K = 10
KEYWORD_K=5
 
 
def search_chunks(question: str, conn, model: SentenceTransformer) -> list[dict]:
    #Convert question to vector
    question_vector = model.encode(question).tolist()

    #Vector similarity search
    with conn.cursor() as cur:
        cur.execute("""
            SELECT section, title, content, page, source
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (question_vector, TOP_K))
        vector_rows = cur.fetchall()

    #Keyword search with synonyms
    synonyms = {
        "enter": ["enter", "entry", "access"],
        "notice": ["notice", "notification"],
        "bond": ["bond", "deposit"],
        "rent": ["rent", "payment"],
        "exit": ["exit", "terminate", "termination", "leave", "vacate", "end"],
        "lease": ["lease", "agreement", "tenancy"],
        "break": ["break", "terminate", "termination", "end", "vacate"],
        "leave": ["leave", "vacate", "terminate", "termination"],
    }

    #Build search keywords including synonyms
    base_keywords = [w for w in question.lower().split() if len(w) >= 4]
    expanded_keywords = set(base_keywords)
    for word in base_keywords:
        if word in synonyms:
            expanded_keywords.update(synonyms[word])

    keyword_rows = []
    with conn.cursor() as cur:
        for keyword in expanded_keywords:
            cur.execute("""
                SELECT section, title, content, page, source
                FROM chunks
                WHERE content ~* %s
                LIMIT %s;
            """, (f'\\m{keyword}\\M', KEYWORD_K))
            keyword_rows.extend(cur.fetchall())

    #Combine both, removing duplicates
    seen = set()
    rows = []
    for row in vector_rows + keyword_rows:
        if row[0] not in seen:
            seen.add(row[0])
            rows.append(row)

    #Convert to list of dictionaries
    results = []
    for row in rows:
        results.append({
            "section": row[0],
            "title": row[1],
            "content": row[2],
            "page": row[3],
            "source": row[4]
        })

    return results
 
#Build the prompt
def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the prompt we send to Claude.
 
    We give Claude:
    - The user's question
    - The relevant sections from the NSW Act
    - Instructions to answer clearly and cite sections
 
    Args:
        question: the user's question
        chunks: the relevant sections retrieved from the database
 
    Returns:
        a formatted prompt string
    """
 
    # Format the retrieved sections into readable text
    context = ""
    for chunk in chunks:
        context += f"Section {chunk['section']} — {chunk['title']}\n"
        context += f"{chunk['content']}\n"
        context += f"[Source: {chunk['source']}, Page {chunk['page']}]\n\n"
 
    # Build the full prompt
    prompt = f"""You are a helpful assistant that answers questions about NSW residential tenancy law.
 
Use ONLY the sections provided below to answer the question. 
Always cite which section your answer comes from.
If the answer is not in the provided sections, say "I couldn't find this in the provided sections."
Keep your answer clear and in plain English — avoid legal jargon where possible.
 
RELEVANT SECTIONS FROM THE NSW RESIDENTIAL TENANCIES ACT:
{context}
 
QUESTION: {question}
 
ANSWER:"""
 
    return prompt
 
 
#Call claude
def call_claude(prompt: str) -> str:
    """
    Send the prompt to Claude and return the answer.
 
    Args:
        prompt: the full prompt including question and context
 
    Returns:
        Claude's answer as a string
    """
 
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
 
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
 
    return message.content[0].text
 
 
#Main RAG function
def answer_question(question: str) -> str:
    """
    The main function that ties everything together.
    Takes a question and returns an answer.
 
    Args:
        question: the user's question
 
    Returns:
        a plain English answer with citations
    """
 
    #Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
 
    #Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Load reranker model
    reranker = load_reranker()
 
    #Search for relevant sections
    print(f"Searching for relevant sections...")
    chunks = search_chunks(question, conn, model)
 
    print(f"Found {len(chunks)} relevant sections:")
    for chunk in chunks:
        print(f"  - Section {chunk['section']}: {chunk['title']}")

    # Step 2: Rerank the results
    print(f"Reranking {len(chunks)} chunks...")
    chunks = rerank(question, chunks, reranker)
    print(f"Top {len(chunks)} chunks after reranking:")
    for chunk in chunks:
        print(f"  - Section {chunk['section']}: {chunk['title']} (score: {chunk['rerank_score']:.3f})")
 
    #Build the prompt
    prompt = build_prompt(question, chunks)
 
    #Call Claude
    print(f"\nCalling Claude...")
    answer = call_claude(prompt)
 
    conn.close()
 
    return answer
 
 

if __name__ == "__main__":
    # Test questions
    test_questions = [
        "when can i exit from lease",
    ]
 
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")
        answer = answer_question(question)

        print(f"\nANSWER:\n{answer}")
        print()