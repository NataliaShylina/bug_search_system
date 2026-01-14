import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import json
from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/bugsearch")

# with engine.connect() as conn:
#     df_bugs = pd.read_sql("SELECT bug_id, summary, description FROM bugs", conn)
query = """
SELECT
    bug_id,
    summary,
    description
FROM bugs
"""

df_bugs = pd.read_sql(query, engine)
print("Liczba bugów do analizy:", len(df_bugs))

# 2. Sentiment analysis
def compute_sentiment(text):
    # if not text or text.strip() == "":
    if text is None or text.strip() == "":
        return 0.0
    return TextBlob(text).sentiment.polarity  # od -1 do 1

df_bugs['sentiment_score'] = df_bugs['description'].apply(compute_sentiment)

# 3. TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_bugs['description'].fillna(''))

# df_bugs['tfidf_vector'] = [json.dumps(row) for row in tfidf_matrix.toarray().tolist()]
df_bugs["tfidf_vector"] = [
    json.dumps(vec.tolist())
    for vec in tfidf_matrix.toarray()
]

# 4. SBERT embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # szybki model, multilingual
# embeddings = model.encode(df_bugs['description'].fillna(''), show_progress_bar=True)
embeddings = model.encode(
    df_bugs["description"].fillna("").tolist(),
    batch_size=32,
    show_progress_bar=True
)

df_bugs['embedding_vector'] = [json.dumps(vec.tolist()) for vec in embeddings]

df_update = df_bugs[
    ["bug_id", "sentiment_score", "tfidf_vector", "embedding_vector"]
]

df_update.to_sql(
    "bugs_nlp_tmp",
    engine,
    if_exists="replace",
    index=False
)

#9 Batch UPDATE w PostgreSQL
with engine.begin() as conn:
    conn.execute(text("""
        UPDATE bugs
        SET
            sentiment_score = tmp.sentiment_score,
            tfidf_vector = tmp.tfidf_vector,
            embedding_vector = tmp.embedding_vector
        FROM bugs_nlp_tmp tmp
        WHERE bugs.bug_id = tmp.bug_id
    """))

with engine.begin() as conn:
    conn.execute(text("DROP TABLE bugs_nlp_tmp"))

check = pd.read_sql("""
SELECT
    COUNT(*) FILTER (WHERE sentiment_score IS NOT NULL) AS sentiment_ok,
    COUNT(*) FILTER (WHERE tfidf_vector IS NOT NULL) AS tfidf_ok,
    COUNT(*) FILTER (WHERE embedding_vector IS NOT NULL) AS embedding_ok
FROM bugs
""", engine)

print(check)

print("NLP na opisach bugów zakończone")
