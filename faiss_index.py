import json
import faiss
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

# KONFIGURACJA
DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/bugsearch"
INDEX_PATH = "faiss_bug_index.bin"
META_PATH = "faiss_bug_meta.csv"

model = SentenceTransformer("all-MiniLM-L6-v2")
engine = create_engine(DB_URL)

print("Wczytywanie embeddingów z PostgreSQL...")

df = pd.read_sql("""
    SELECT 
        bug_id, 
        summary, 
        priority, 
        status, 
        resolution,
        created_at, 
        resolved_at,
        description,
        sentiment_score, 
        embedding_vector
    FROM bugs
""", engine)

# Dedupikacja
df = df.drop_duplicates(subset=["bug_id"]).reset_index(drop=True)

# OBLICZANIE SENTYMENTU DLA KAŻDEGO WIERSZA ---
print("Obliczanie sentymentu (Summary + Description)...")
def calculate_sentiment(row):
    # Łączę summary i description do analizy emocji
    # full_text = str(row['summary']) + " " + str(row['description'] if row['description'] else "")
    summary = str(row['summary']) if pd.notnull(row['summary']) else ""
    description = str(row['description']) if pd.notnull(row['description']) else ""
    full_text = (summary + " " + description).strip()

    if not full_text:
        return 0.0
    return TextBlob(full_text).sentiment.polarity

df['sentiment_score'] = df.apply(calculate_sentiment, axis=1)
print("Generowanie nowych embeddingów (Summary + Description)...")

# 2. Tworzę tekst do embeddingów
df["text_for_embedding"] = df["summary"].fillna("") + " " + df["description"].fillna("")

# 3. Generuję wektory na podstawie połączonego tekstu
embeddings = model.encode(df["text_for_embedding"].tolist()).astype("float32")
faiss.normalize_L2(embeddings) # Normalizacja (wymagana dla cosinus)

print("Liczba embeddingów:", embeddings.shape[0])
print("Wymiar embeddingu:", embeddings.shape[1])

# FAISS INDEX
print("Budowa indeksu FAISS...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosinus po normalizacji
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

# Zapis metadanych (mapowanie indeks → bug)
meta_columns = [
    "bug_id", "summary", "priority", "status",
    "resolution", "created_at", "resolved_at",
    "sentiment_score", "description"

]
df[meta_columns].to_csv(META_PATH, index=False)

print("Indeks FAISS zapisany:", INDEX_PATH)
print("Metadane zapisane:", META_PATH)
print("FAISS index gotowy")
