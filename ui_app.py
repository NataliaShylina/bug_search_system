import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_bug_index.bin"
META_PATH = "faiss_bug_meta.csv"

app = FastAPI(title="Bug Semantic Search")

templates = Jinja2Templates(directory="templates")

print("Ładowanie FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Ładowanie metadanych...")
meta = pd.read_csv(META_PATH)

print("Ładowanie modelu SBERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "results": [],
            "query": "",
            "top_k": 5,
            "sort_by": "score",
            "error": None
        }
    )


@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(5),
    sort_by: str = Form("score")
):
    results = []
    error = None

    try:
        query_vec = model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = index.search(query_vec, top_k)

        for score, idx in zip(scores[0], indices[0]):
            bug = meta.iloc[idx]
            results.append({
                "bug_id": int(bug["bug_id"]),
                "summary": bug["summary"],
                "score": float(score)
            })

        if sort_by == "bug_id":
            results = sorted(results, key=lambda x: x["bug_id"])
        else:
            results = sorted(results, key=lambda x: x["score"], reverse=True)

    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "results": results,
            "query": query,
            "top_k": top_k,
            "sort_by": sort_by,
            "error": error
        }
    )
