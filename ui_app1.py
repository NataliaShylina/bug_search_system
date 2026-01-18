import faiss
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import json
from sqlalchemy import create_engine

INDEX_PATH = "faiss_bug_index.bin"
META_PATH = "faiss_bug_meta.csv"
DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/bugsearch"
engine = create_engine(DB_URL)

app = FastAPI(title="Bug Semantic Search")
templates = Jinja2Templates(directory="templates")

# Ładowanie zasobów
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
meta = pd.read_csv(META_PATH)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("search1.html", {
        "request": request, "results": [], "query": "", "top_k": 1000, "error": None
    })

@app.post("/search", response_class=HTMLResponse)
async def search(
        request: Request,
        query: str = Form(...),
        top_k: int = Form(1000),
        sort_by: str = Form("score"),
        sort_order: str = Form("desc"),
        priority_filter: str = Form("all"),
        status_filter: str = Form("all")
):
    is_ascending = (sort_order == "asc")

    try:
        # 1. Szukanie semantyczne w FAISS
        query_vec = model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, 1000)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(meta) and score < 100.0:
                bug = meta.iloc[idx].to_dict()
                bug["score"] = float(score)
                results.append(bug)

        # 2. Filtrowanie (Pandas style)
        df_res = pd.DataFrame(results)

        if priority_filter != "all":
            df_res = df_res[df_res['priority'] == priority_filter]
        if status_filter != "all":
            df_res = df_res[df_res['status'] == status_filter]

        # 3. Sortowanie
        if sort_by == "score":
            df_res = df_res.sort_values(by="score", ascending=is_ascending)
        elif sort_by == "status":
            status_order = {
                'To Do': 0,
                'In Progress': 1,
                'Code review': 2,
                'On review': 3,
                'READY FOR QA': 4,
                'QA REVIEW': 5,
                'Done': 6,
                'BLOCKED': 7
            }
            df_res['status_val'] = df_res['status'].map(status_order).fillna(99)
            df_res = df_res.sort_values(by="status_val", ascending=is_ascending).drop('status_val', axis=1)
        elif sort_by == "sentiment":
            df_res = df_res.sort_values(by="sentiment_score", ascending=is_ascending)
        elif sort_by == "resolution":
            df_res = df_res.sort_values(by="resolution", ascending=True)
        elif sort_by == "summary":
            df_res = df_res.sort_values(by="summary", ascending=True)
        elif sort_by == "priority":
            priority_order = {'Highest': 0,
                              'High': 1,
                              'Medium': 2,
                              'Low': 3,
                              'Lowest': 4}
            df_res['prio_val'] = df_res['priority'].map(priority_order)
            df_res = df_res.sort_values(by="prio_val", ascending=is_ascending).drop('prio_val', axis=1)
        else:
            # Sortowanie dla status, resolution, summary
            col = "sentiment_score" if sort_by == "sentiment" else sort_by
            df_res = df_res.sort_values(by=sort_by, ascending=is_ascending)

            # 4. Wybór finalnych top_k
        final_results = df_res.head(top_k).to_dict(orient="records")

        return templates.TemplateResponse("search1.html", {
            "request": request,
            "results": final_results,
            "query": query,
            "top_k": top_k,
            "sort_by": sort_by,
            "priority_filter": priority_filter,
            "status_filter": status_filter,
            "error": None
        })

    except Exception as e:
        return templates.TemplateResponse("search1.html", {
            "request": request,
            "error": str(e),
            "results": [],
            "query": query,
            "top_k": top_k
        })


@app.post("/upload")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    global meta, index
    try:
        df_new = pd.read_csv(file.file)

        # Obliczanie brakujących danych "w locie"
        text_to_embed = df_new['summary'].fillna("").astype(str) + " " + df_new['description'].fillna("").astype(str)
        text_list = text_to_embed.tolist()

        # Sentyment
        df_new['sentiment_score'] = df_new.apply(
            lambda row: TextBlob(
                str(row['description'] if pd.notnull(row['description']) else row['summary'])).sentiment.polarity,
            axis=1
        )
        # Embeddingi do FAISS
        new_vectors = model.encode(text_list).astype("float32")
        faiss.normalize_L2(new_vectors)

        # 3. Zapis do PostgreSQL
        # embedding_vector musi być zapisany jako JSON string
        df_to_db = df_new.copy()
        df_to_db['embedding_vector'] = [json.dumps(v.tolist()) for v in new_vectors]
        df_to_db.to_sql('bugs', con=engine, if_exists='append', index=False)

        # Aktualizacja indeksu FAISS
        index.add(new_vectors)
        faiss.write_index(index, INDEX_PATH)

        # Aktualizacja metadanych CSV
        meta = pd.concat([meta, df_new], ignore_index=True)
        meta.to_csv(META_PATH, index=False)

        return templates.TemplateResponse("search1.html", {
            "request": request,
            "message": f"Successfully added {len(df_new)} bugs!",
            "query": "",
            "top_k": 10,
            "results": [],
            "sort_by": "score",
            "priority_filter": "all",
            "status_filter": "all"
        })

    except Exception as e:
        return templates.TemplateResponse("search1.html", {
            "request": request,
            "error": f"Upload failed: {str(e)}",
            "query": "", "top_k": 10, "results": []
        })