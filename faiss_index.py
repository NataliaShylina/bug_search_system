import json
import faiss
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# KONFIGURACJA
DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/bugsearch"
INDEX_PATH = "faiss_bug_index.bin"
META_PATH = "faiss_bug_meta.csv"

engine = create_engine(DB_URL)

print("Wczytywanie embeddingów z PostgreSQL...")

df = pd.read_sql("""
    SELECT bug_id, summary, embedding_vector
    FROM bugs
    WHERE embedding_vector IS NOT NULL
""", engine)

# Dedupikacja
df = df.drop_duplicates(subset=["bug_id"]).reset_index(drop=True)

# Parsowanie JSON
df["embedding_vector"] = df["embedding_vector"].apply(json.loads)

# Macierz embeddingów
embeddings = np.vstack(df["embedding_vector"].values).astype("float32")

print("Liczba embeddingów:", embeddings.shape[0])
print("Wymiar embeddingu:", embeddings.shape[1])

# FAISS INDEX
print("Budowa indeksu FAISS...")

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosinus po normalizacji

# Normalizacja (wymagana dla cosinus)
faiss.normalize_L2(embeddings)

index.add(embeddings)

# Zapis indeksu
faiss.write_index(index, INDEX_PATH)

# Zapis metadanych (mapowanie indeks → bug)
df[["bug_id", "summary"]].to_csv(META_PATH, index=False)

print("Indeks FAISS zapisany:", INDEX_PATH)
print("Metadane zapisane:", META_PATH)
print("FAISS index gotowy")
