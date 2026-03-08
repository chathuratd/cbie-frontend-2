"""
api/main.py
===========
FastAPI application entry point for the CBIE Microservice.

Start the server:
    cd cbie_engine
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Interactive API docs:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""
from __future__ import annotations
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the cbie_engine root is on the Python path so that internal imports
# like `from pipeline import CBIEPipeline` resolve correctly when the app is
# launched from within the api/ sub-package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import get_logger
from api.dependencies import init_pipeline
from api.routers import context, profiles
from api.routers import pipeline_router
from api.routers import admin
from api.models import HealthResponse, RootResponse

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------
# The pipeline singleton (with its heavy BART model) is loaded ONCE here on
# startup so that every request can reuse it without re-initialising 1.5 GB.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources on startup, clean up on shutdown."""
    log.info("CBIE Microservice starting up", extra={"stage": "STARTUP"})
    init_pipeline()   # Loads BART NLP model + Azure OpenAI client
    log.info("All resources ready — accepting requests", extra={"stage": "STARTUP"})
    yield
    # Shutdown cleanup (nothing explicit needed for now)
    log.info("CBIE Microservice shutting down", extra={"stage": "SHUTDOWN"})


# ---------------------------------------------------------------------------
# App Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Core Behaviour Identification Engine (CBIE) API",
    description=(
        "Microservice API for the CBIE — an offline batch-processing analytical "
        "engine that analyses user interaction histories to identify stable "
        "long-term core behaviours and synthesise them into an Identity Anchor "
        "Prompt for LLM personalisation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
# Allowing all origins for now; tighten this once the frontend domain is known.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Include Routers
# ---------------------------------------------------------------------------

app.include_router(context.router)
app.include_router(pipeline_router.router)
app.include_router(profiles.router)
app.include_router(admin.router)


# ---------------------------------------------------------------------------
# Root & Health Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    response_model=RootResponse,
    tags=["Service Info"],
    summary="API Root",
)
async def root():
    """Returns basic service information and links to the documentation."""
    return RootResponse()


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service Info"],
    summary="Health Check",
    description=(
        "Standard health check for container orchestration (Docker, Kubernetes). "
        "Returns `pipeline_ready: true` only after the heavy NLP model has loaded."
    ),
)
async def health():
    """Health check endpoint — confirms the pipeline singleton is ready."""
    from api.dependencies import _pipeline_instance
    return HealthResponse(
        status="ok",
        engine="CBIE",
        version="1.0.0",
        pipeline_ready=_pipeline_instance is not None,
    )
