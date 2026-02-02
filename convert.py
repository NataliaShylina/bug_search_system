# import pandas as pd
# from sqlalchemy import create_engine
#
# engine = create_engine(
#     "postgresql+psycopg2://postgres:HASLO@localhost:5432/bugsearch"
# )
#
# df = pd.read_csv("GFG_projects.csv")
#
# print(df.head())
# print("Liczba rekordów:", len(df))
#
# df.to_sql("bugs", engine, if_exists="append", index=False)
#
# print("ZAPIS DO BAZY ZAKOŃCZONY")

#Wymiar wektorów SBERT
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(model.get_sentence_embedding_dimension())

#Liczba wektorów w indeksie FAISS
import faiss
print(index.ntotal)

#Czas wyszukiwania
import time

query = "crash when saving file"
q_emb = model.encode([query]).astype("float32")
faiss.normalize_L2(q_emb)

start = time.time()
D, I = index.search(q_emb, k=10)
end = time.time()

print("Czas wyszukiwania:", (end - start) * 1000, "ms")


