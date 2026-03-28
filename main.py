import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.dependencies import lifespan
from routers import adjacency, cache, chains, classifier, events, latencies, search, state_detectors, stats

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events.router)
app.include_router(adjacency.router)
app.include_router(classifier.router)
app.include_router(cache.router)
app.include_router(chains.router)
app.include_router(latencies.router)
app.include_router(search.router)
app.include_router(state_detectors.router)
app.include_router(stats.router)

# Serve Angular UI in production — static assets first, then SPA catch-all
_ui_dist = os.path.join(os.path.dirname(__file__), "ui", "chain-search", "dist", "chain-search", "browser")
if os.path.isdir(_ui_dist):

    @app.get("/claude-api/{full_path:path}")
    async def serve_ui(full_path: str = ""):
        file_path = os.path.join(_ui_dist, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_ui_dist, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
