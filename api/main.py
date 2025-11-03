"""FastAPI server for Pokemon generation API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1 import endpoint

app = FastAPI(
    title="Pokemon Generator API",
    description="Generate Pokemon images from text prompts using a Conditional VAE",
    version="1.0.0"
)

# Configure CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Angular dev server
        "http://localhost:8080",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 routes
app.include_router(endpoint.router, prefix="/api/v1", tags=["Pokemon Generation"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Pokemon Generator API is running",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/v1/generate",
            "health": "/api/v1/health"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": endpoint.model is not None,
        "device": str(endpoint.device) if endpoint.device else "unknown"
    }
