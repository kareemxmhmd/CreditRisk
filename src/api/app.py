"""
FastAPI Main Application Entrypoint for CreditRisk Engine.
Includes Correlation ID tracing, CORS middleware, and lifecycle management.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import router, scoring_service
from src.config import MODEL_VERSION, REQUEST_CORRELATION_HEADER
from src.monitoring.telemetry import TelemetryService

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get(REQUEST_CORRELATION_HEADER)
        if not correlation_id:
            correlation_id = TelemetryService.generate_correlation_id()

        response = await call_next(request)
        response.headers[REQUEST_CORRELATION_HEADER] = correlation_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts into memory at startup
    try:
        scoring_service.load()
        logger.info("Enterprise model artifacts and feature store loaded successfully at startup.")
    except Exception as e:
        logger.warning(f"Warning: Could not eagerly load models at startup: {e}")
    yield


app = FastAPI(
    title="CreditRisk — Enterprise AI Loan Decisioning Engine",
    description=(
        "Enterprise-scale, cost-sensitive credit risk evaluation service with "
        "calibrated default probabilities, automated 3-tier decisions (Approve/Refer/Reject), "
        "Champion/Challenger Canary & Shadow routing, Feature Store entity hydration, "
        "Adverse Action reason codes (SHAP), fair lending auditing, and Prometheus telemetry."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CorrelationIdMiddleware)

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
        "service": "CreditRisk Enterprise AI Decisioning Engine",
        "status": "ONLINE",
        "version": "2.0.0",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "prometheus_metrics_url": "/api/v1/metrics/prometheus",
        "model_version": MODEL_VERSION,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
