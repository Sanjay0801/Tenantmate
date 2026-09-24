import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import psycopg2
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
import anthropic

try:
    from src.rag import search_chunks, build_prompt, call_claude
    from src.reranker import load_reranker, rerank
except ModuleNotFoundError:
    from rag import search_chunks, build_prompt, call_claude
    from reranker import load_reranker, rerank
 
# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()
 
# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}
 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
 
 
# ── State — what gets passed between nodes ─────────────────────────────────────
class AgentState(TypedDict):
    """
    The state that flows through the LangGraph agent.
    Each node reads from and writes to this state.
    """
    question: str           
    tool: str               # which tool the router chose
    chunks: list           
    calculation: str        
    answer: str             # final answer
 
 
# ── Load models once ───────────────────────────────────────────────────────────
print("Loading models...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
reranker_model = load_reranker()
conn = psycopg2.connect(**DB_CONFIG)
print("Models loaded!")
 
 
# ── Node 1: Router — decides which tool to use ────────────────────────────────
def router_node(state: AgentState) -> AgentState:
  
    question = state["question"].lower()
 
    # Check if it's a calculation question
    # looks for dollar signs, numbers with $, or words like "how much"
    if re.search(r'\$\d+|\d+\s*dollar|\d+\s*per\s*week|how much|calculate|bond amount', question):
        tool = "calculator"
 
    # Check if it's a direct section lookup
    # looks for "section 55", "s55", "sec 55" etc
    elif re.search(r'section\s+\d+|what does section|s\d+\b|sec\s+\d+', question):
        tool = "lookup"
 
    # Everything else uses search
    else:
        tool = "search"
 
    print(f"Router decision: {tool}")
    return {**state, "tool": tool}
 
 
# ── Node 2: Search tool ────────────────────────────────────────────────────────
def search_node(state: AgentState) -> AgentState:
   
    print(f"Using search tool...")
    chunks = search_chunks(state["question"], conn, embed_model)
    chunks = rerank(state["question"], chunks, reranker_model)
    return {**state, "chunks": chunks}
 
 
# ── Node 3: Calculator tool ────────────────────────────────────────────────────
def calculator_node(state: AgentState) -> AgentState:
  
    print(f"Using calculator tool...")
    question = state["question"]
 
    # Extract dollar amount from question
    # looks for patterns like "$500", "500 per week", "500/week"
    amount_match = re.search(r'\$(\d+(?:\.\d{2})?)|(\d+(?:\.\d{2}?)?)\s*(?:per week|/week|a week|\$)|(\d+(?:\.\d{2}?)?)\s*dollar', question)
 
    calculation = ""
 
    if amount_match:
        # Get the weekly rent amount
        amount = float(amount_match.group(1) or amount_match.group(2) or amount_match.group(3))
 
        # Calculate common tenancy amounts
        bond_4_weeks = amount * 4
        bond_2_weeks = amount * 2
        monthly = amount * 52 / 12
 
        calculation = f"""
Based on a weekly rent of ${amount:.2f}:
 
Calculations:
- Maximum bond (4 weeks rent): ${bond_4_weeks:.2f}
- 2 weeks rent: ${bond_2_weeks:.2f}
- Monthly equivalent: ${monthly:.2f}
 

        """.strip()
    else:
        calculation = "I couldn't find a specific dollar amount in your question. Please specify the weekly rent amount, e.g. 'My rent is $500/week'."
 
    # Also search for relevant sections to add context
    chunks = search_chunks("bond maximum amount rent", conn, embed_model)
    chunks = rerank("bond maximum amount rent", chunks, reranker_model)
 
    return {**state, "calculation": calculation, "chunks": chunks}
 
 
# ── Node 4: Lookup tool ────────────────────────────────────────────────────────
def lookup_node(state: AgentState) -> AgentState:
    """
    Direct section lookup by number.
    Much faster than search for questions like "What does Section 55 say?"
    """
    print(f"Using lookup tool...")
    question = state["question"]
 
    # Extract section number from question
    section_match = re.search(r'section\s+(\d+[A-Z]?)|s(\d+[A-Z]?)\b|sec\s+(\d+[A-Z]?)', question, re.IGNORECASE)
 
    chunks = []
 
    if section_match:
        section_num = (section_match.group(1) or section_match.group(2) or section_match.group(3)).upper()
        print(f"Looking up Section {section_num}...")
 
        with conn.cursor() as cur:
            cur.execute("""
                SELECT section, title, content, page, source
                FROM chunks
                WHERE section = %s;
            """, (section_num,))
            row = cur.fetchone()
 
        if row:
            chunks = [{
                "section": row[0],
                "title": row[1],
                "content": row[2],
                "page": row[3],
                "source": row[4]
            }]
            print(f"Found Section {section_num}: {row[1]}")
        else:
            print(f"Section {section_num} not found in database")
            # Fall back to search
            chunks = search_chunks(question, conn, embed_model)
            chunks = rerank(question, chunks, reranker_model)
    else:
        # Fall back to search if no section number found
        chunks = search_chunks(question, conn, embed_model)
        chunks = rerank(question, chunks, reranker_model)
 
    return {**state, "chunks": chunks}
 
 
# ── Node 5: Answer generator ───────────────────────────────────────────────────
def answer_node(state: AgentState) -> AgentState:
    """
    Generates the final answer using Claude.
    Handles all 3 tool types differently.
    """
    print(f"Generating answer...")
 
    # Calculator tool — prepend calculation to the answer
    if state["tool"] == "calculator" and state.get("calculation"):
        # First show calculation, then add legal context
        prompt = build_prompt(state["question"], state["chunks"])
        legal_context = call_claude(prompt)
 
        answer = f"{state['calculation']}\n\n---\n\n**Legal Context:**\n{legal_context}"
 
    # Search or lookup — standard RAG answer
    else:
        prompt = build_prompt(state["question"], state["chunks"])
        answer = call_claude(prompt)
 
    return {**state, "answer": answer}
 
 
# ── Router function — tells LangGraph which node to go to next ─────────────────
def route_to_tool(state: AgentState) -> str:
    """
    After the router node runs, this function tells LangGraph
    which node to go to next based on the tool decision.
    """
    return state["tool"]
 
 
# ── Build the LangGraph ────────────────────────────────────────────────────────
def build_agent():

    # Create the graph with our state
    graph = StateGraph(AgentState)
 
    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("search", search_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("answer", answer_node)
 
    # Set the entry point
    graph.set_entry_point("router")
 
    # Add conditional edges from router to tools
    graph.add_conditional_edges(
        "router",
        route_to_tool,
        {
            "search": "search",
            "calculator": "calculator",
            "lookup": "lookup"
        }
    )
 
    # All tools go to answer node
    graph.add_edge("search", "answer")
    graph.add_edge("calculator", "answer")
    graph.add_edge("lookup", "answer")
 
    # Answer node goes to END
    graph.add_edge("answer", END)
 
    # Compile and return
    return graph.compile()
 
 
# ── Main function ──────────────────────────────────────────────────────────────
def answer_question_with_agent(question: str) -> str:
    """
    Main function — replaces answer_question() from rag.py.
    Uses the LangGraph agent instead of the linear pipeline.
    """
    agent = build_agent()
 
    # Run the agent
    result = agent.invoke({
        "question": question,
        "tool": "",
        "chunks": [],
        "calculation": "",
        "answer": ""
    })
 
    return result["answer"]


 # ── Test it directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "Can my landlord enter without notice?",           # → search tool
        "My rent is $500/week, how much bond can be charged?",  # → calculator tool
        "What does Section 55 say?",                       # → lookup tool
    ]
 
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")
        answer = answer_question_with_agent(question)
        print(f"\nANSWER:\n{answer}")
        print()