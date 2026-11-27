
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from pydantic import BaseModel, Field
from pathlib import Path
import os

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from search.search_engine import SearchEngine
from frontend.summarizer import summarize_results

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / "artifacts" / "index"))
WEB_DIR = Path(__file__).resolve().parent

# the search engine shares one index file handle across requests
search_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the index once instead of loading it for every query
    with SearchEngine(INDEX_DIR) as engine:
        app.state.engine = engine
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/search")
def search(q: str, limit: int = Query(default=5, ge=1, le=100)):
    # serialize searches since seek() modifies the shared file position
    with search_lock:
        start_time = time.perf_counter()
        urls, total = app.state.engine.search(q, limit)
        latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "query": q,
        "total": total,
        "latency_ms": round(latency_ms, 4),
        "results": [
            {"title": url, "url": url, "description": ""}
            for url in urls
        ]
    }


class SummaryRequest(BaseModel):
    query: str = Field(max_length=500)
    urls: list[str] = Field(max_length=5)


@app.post("/api/summarize")
async def summarize(request: SummaryRequest):
    summary = await summarize_results(request.query, request.urls)
    return {"summary": summary}


# serve the existing html/css/js from the same server
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="frontend")
