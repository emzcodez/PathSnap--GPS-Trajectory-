"""PathSnap Backend Server Entrypoint

Joint Ownership: Prateek & Priyansh
Hackathon: PathSnap Distributed Map-Matching

Runs FastAPI backend on Laptop 1 to process evaluation requests from Laptop 2.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router as api_router

app = FastAPI(
    title="PathSnap Map-Matching Backend API",
    description="Distributed map-matching and sequential route reconstruction API",
    version="1.0.0"
)

# Enable CORS for two-laptop deployment architecture (Laptop 1 API <-> Laptop 2 UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root():
    return {
        "system": "PathSnap Distributed Map-Matching API",
        "checkpoint": "First-Phase Sprint",
        "docs_url": "/docs",
        "api_health": "/api/health",
        "trips_url": "/api/trips"
    }


if __name__ == "__main__":
    uvicorn.run("PathSnap.backend.main:app", host="0.0.0.0", port=8000, reload=False)
