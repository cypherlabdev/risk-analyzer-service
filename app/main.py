"""Risk Analyzer Service - ML-based fraud detection"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import logging

from app.models.risk_models import OrderRiskRequest, RiskScore, get_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
risk_analysis_total = Counter(
    'risk_analysis_total',
    'Total risk analyses performed',
    ['risk_level']
)
risk_score_histogram = Histogram(
    'risk_score',
    'Distribution of risk scores'
)
analysis_duration = Histogram(
    'analysis_duration_seconds',
    'Risk analysis duration'
)

# Create FastAPI app
app = FastAPI(
    title="Risk Analyzer Service",
    description="ML-based fraud detection and risk analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "risk-analyzer"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/analyze", response_model=RiskScore)
async def analyze_risk(request: OrderRiskRequest):
    """Analyze risk for an order"""
    try:
        logger.info(f"Analyzing risk for user {request.user_id}, stake: {request.stake}")

        # Get analyzer
        analyzer = get_analyzer()

        # Perform analysis
        with analysis_duration.time():
            result = analyzer.analyze(request)

        # Update metrics
        risk_analysis_total.labels(risk_level=result.risk_level).inc()
        risk_score_histogram.observe(result.score)

        logger.info(
            f"Risk analysis complete: user={request.user_id}, "
            f"score={result.score:.3f}, level={result.risk_level}"
        )

        return result

    except Exception as e:
        logger.error(f"Risk analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "risk-analyzer",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)
