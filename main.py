from core.dependencies import lifespan
from fastapi import FastAPI

from routers import adjacency, cache, chains, classifier, events

app = FastAPI(lifespan=lifespan)

app.include_router(events.router)
app.include_router(adjacency.router)
app.include_router(classifier.router)
app.include_router(cache.router)
app.include_router(chains.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
