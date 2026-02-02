import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/bugsearch"
)

df = pd.read_sql("""
    SELECT bug_id, summary, description, tfidf_vector, embedding_vector, sentiment_score
    FROM bugs
    WHERE tfidf_vector IS NOT NULL
      AND embedding_vector IS NOT NULL
""", engine)

df = df.drop_duplicates(subset=["bug_id"]).reset_index(drop=True)

df["tfidf_vector"] = df["tfidf_vector"].apply(json.loads)
df["embedding_vector"] = df["embedding_vector"].apply(json.loads)

tfidf_matrix = np.vstack(df["tfidf_vector"].values)
embedding_matrix = np.vstack(df["embedding_vector"].values)

# Model SBERT
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_vectorizer.fit(df['description'].fillna(''))

# WYSZUKIWANIE
def search_bugs(query: str, top_k: int = 10):
    """
    Zwraca top_k najbardziej podobnych bugów
    na podstawie TF-IDF i embeddingów
    """

    # Embedding zapytania
    query_embedding = sbert_model.encode([query])
    embedding_scores = cosine_similarity(
        query_embedding, embedding_matrix
    )[0]

    query_tfidf = tfidf_vectorizer.transform([query])
    # TF-IDF zapytania = średnia TF-IDF słów
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    #     embedding_scores.reshape(1, -1),
    #     embedding_scores.reshape(1, -1)
    # )[0]

    # Łączny ranking
    final_score = 0.7 * embedding_scores + 0.3 * tfidf_scores

    results = df.copy()
    # results["score"] = final_score
    results["score_semantic"] = embedding_scores  # Miara z AI
    results["score_tfidf"] = tfidf_scores  # Miara ze słów
    results["similarity_score"] = final_score  # Wynik końcowy (ważony)

    return results.sort_values("similarity_score", ascending=False).head(top_k)

if __name__ == "__main__":
    res = search_bugs(
        "authentication error when cloning repository",
        top_k=5
    )

    # print(res[["bug_id", "summary", "score"]])
    print("\n--- TOP 5 NAJLEPSZYCH DOPASOWAŃ ---")
    print(res[[
        "bug_id",
        "summary",
        "sentiment_score",  # Wyciągnięte z bazy
        "score_semantic",   # Cosinus SBERT
        "score_tfidf",      # Cosinus TF-IDF
        "similarity_score"  # Końcowy ranking
    ]])