"""
FastAPI Main Application Entrypoint for CreditRisk Engine.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, scoring_service
from src.config import MODEL_VERSION

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts into memory at startup
    try:
        scoring_service.load()
        logger.info("Model artifacts loaded successfully at startup.")
    except Exception as e:
        logger.warning(f"Warning: Could not eagerly load models at startup: {e}")
    yield


app = FastAPI(
    title="CreditRisk — AI-Powered Loan Decisioning Engine",
    description=(
        "Production-grade, cost-sensitive credit risk evaluation service with "
        "automated 3-tier decisions (Approve/Refer/Reject), risk tiering, "
        "Adverse Action reason codes (SHAP), fair lending auditing, and drift monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Web UI and microservices integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "service": "CreditRisk AI Decisioning Engine",
        "status": "ONLINE",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "model_version": MODEL_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
