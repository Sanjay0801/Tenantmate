import json
import psycopg2
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

CHUNKS_PATH = Path("data/processed/chunks.json")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

def load_chunks(chunks_path: Path) -> list[dict]:
    """Read the chunks.json file and return a list of section dictionaries."""
    print(f"Loading chunks from {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks
 
 
#Set up the database table
def setup_database(conn) -> None:
    """
    Create the chunks table in PostgreSQL if it doesn't exist.
    
    The table stores:
    - section number and title
    - the full text content
    - page number and source
    - the vector embedding (384 dimensions for bge-small-en-v1.5)
    """
    with conn.cursor() as cur:
        # Enable pgvector extension (safe to run even if already enabled)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
 
        # Create the table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          SERIAL PRIMARY KEY,
                section     TEXT,
                title       TEXT,
                content     TEXT,
                page        INTEGER,
                source      TEXT,
                embedding   vector(384)
            );
        """)
 
        # Create an index for fast similarity search
        # ivfflat is the standard index type for pgvector
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
 
        conn.commit()
    print("Database table and index ready")
 
 
# Embed chunks and store in database 
def embed_and_store(chunks: list[dict], conn) -> None:
    """
    Convert each chunk into a vector and store in PostgreSQL.
    
    Steps:
    1. Load the embedding model (downloads on first run)
    2. Loop through each chunk
    3. Convert the chunk's content to a vector
    4. Insert into the database
    """
 
    # Load the local embedding model
    # This downloads the model on first run (~90MB)
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded")
 
    # Clear existing data so we don't get duplicates if we re-run
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE chunks;")
        conn.commit()
    print("Cleared existing chunks from database")
 
    # Loop through each chunk, embed it, and store it
    print(f"Embedding and storing {len(chunks)} chunks...")
 
    with conn.cursor() as cur:
        for chunk in tqdm(chunks):
            # Convert the content field to a vector
            # content = "Section 54 — Landlord must not interfere...\n\nA landlord..."
            embedding = model.encode(chunk["content"])
 
            # Convert numpy array to a list for PostgreSQL
            embedding_list = embedding.tolist()
 
            # Insert into database
            cur.execute("""
                INSERT INTO chunks (section, title, content, page, source, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                chunk["section"],
                chunk["title"],
                chunk["content"],
                chunk["page"],
                chunk["source"],
                embedding_list
            ))
 
        # Save all inserts at once
        conn.commit()
 
    print(f"Successfully stored {len(chunks)} chunks in the database")
 
 
#Main 
def main():
    """Run the full pipeline: load chunks → setup DB → embed and store."""
 
    # Load chunks from JSON
    chunks = load_chunks(CHUNKS_PATH)
 
    # Connect to PostgreSQL
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("Connected!")
 
    # Set up the database table
    setup_database(conn)
 
    # Embed and store
    embed_and_store(chunks, conn)
 
    # Verify — count how many rows are in the table
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks;")
        count = cur.fetchone()[0]
        print(f"\nVerification: {count} chunks in database")
 
    # Show a sample from the database
    with conn.cursor() as cur:
        cur.execute("SELECT section, title, page FROM chunks LIMIT 3;")
        rows = cur.fetchall()
        print("\n── Sample rows from database ──")
        for row in rows:
            print(f"Section {row[0]} — {row[1]} (page {row[2]})")
 
    conn.close()
    print("\nDone! Database is ready for search.")
 
 
if __name__ == "__main__":
    main()