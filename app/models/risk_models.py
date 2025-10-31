"""Risk analysis models and ML pipeline"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib


class OrderRiskRequest(BaseModel):
    """Request for risk analysis of an order"""
    user_id: str
    event_id: str
    market_id: str
    stake: Decimal
    odds: Decimal
    user_history: Optional[Dict] = None


class RiskScore(BaseModel):
    """Risk analysis result"""
    score: float = Field(..., ge=0.0, le=1.0, description="Risk score 0-1")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    factors: Dict[str, float] = Field(default_factory=dict)
    recommendation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskAnalyzer:
    """ML-based risk analyzer for fraud detection"""

    def __init__(self):
        # Load pre-trained models (in production, load from storage)
        try:
            self.fraud_model = joblib.load("models/fraud_detector.pkl")
            self.anomaly_model = joblib.load("models/anomaly_detector.pkl")
        except FileNotFoundError:
            # Initialize new models if not found
            self.fraud_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)

        # Risk thresholds
        self.thresholds = {
            "LOW": 0.3,
            "MEDIUM": 0.6,
            "HIGH": 0.8,
            "CRITICAL": 0.95
        }

    def analyze(self, request: OrderRiskRequest) -> RiskScore:
        """Analyze order risk using ML models"""

        # Extract features
        features = self._extract_features(request)

        # Run fraud detection
        fraud_score = self._detect_fraud(features)

        # Run anomaly detection
        anomaly_score = self._detect_anomaly(features)

        # Combine scores
        combined_score = (fraud_score * 0.7) + (anomaly_score * 0.3)

        # Determine risk level
        risk_level = self._determine_risk_level(combined_score)

        # Generate recommendation
        recommendation = self._generate_recommendation(risk_level, combined_score)

        return RiskScore(
            score=combined_score,
            risk_level=risk_level,
            factors={
                "fraud_probability": fraud_score,
                "anomaly_score": anomaly_score,
                "stake_risk": self._calculate_stake_risk(request.stake),
                "odds_risk": self._calculate_odds_risk(request.odds),
            },
            recommendation=recommendation
        )

    def _extract_features(self, request: OrderRiskRequest) -> np.ndarray:
        """Extract ML features from order request"""

        features = []

        # Stake features
        stake_float = float(request.stake)
        features.append(stake_float)
        features.append(np.log1p(stake_float))  # Log transform

        # Odds features
        odds_float = float(request.odds)
        features.append(odds_float)
        features.append(1.0 / odds_float if odds_float > 0 else 0.0)  # Implied probability

        # User history features (if available)
        if request.user_history:
            features.append(request.user_history.get("total_bets", 0))
            features.append(request.user_history.get("win_rate", 0.0))
            features.append(request.user_history.get("avg_stake", 0.0))
            features.append(request.user_history.get("days_active", 0))
        else:
            features.extend([0.0, 0.0, 0.0, 0])  # Defaults for new users

        return np.array(features).reshape(1, -1)

    def _detect_fraud(self, features: np.ndarray) -> float:
        """Detect fraud probability using supervised model"""
        try:
            # Predict fraud probability
            fraud_prob = self.fraud_model.predict_proba(features)[0][1]
            return float(fraud_prob)
        except Exception:
            # Fallback: simple heuristic
            return 0.5

    def _detect_anomaly(self, features: np.ndarray) -> float:
        """Detect anomalies using unsupervised model"""
        try:
            # Anomaly score (-1 for outliers, 1 for inliers)
            score = self.anomaly_model.score_samples(features)[0]
            # Normalize to 0-1 (lower score = more anomalous)
            normalized = 1.0 / (1.0 + np.exp(score))
            return float(normalized)
        except Exception:
            return 0.5

    def _calculate_stake_risk(self, stake: Decimal) -> float:
        """Calculate risk based on stake amount"""
        stake_float = float(stake)

        if stake_float > 10000:
            return 0.9  # Very high stakes
        elif stake_float > 5000:
            return 0.7
        elif stake_float > 1000:
            return 0.5
        elif stake_float > 100:
            return 0.3
        else:
            return 0.1

    def _calculate_odds_risk(self, odds: Decimal) -> float:
        """Calculate risk based on odds value"""
        odds_float = float(odds)

        if odds_float > 100:
            return 0.9  # Extremely high odds (long shot)
        elif odds_float > 50:
            return 0.7
        elif odds_float > 10:
            return 0.5
        elif odds_float < 1.1:
            return 0.6  # Suspiciously low odds
        else:
            return 0.2

    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score"""
        if score >= self.thresholds["CRITICAL"]:
            return "CRITICAL"
        elif score >= self.thresholds["HIGH"]:
            return "HIGH"
        elif score >= self.thresholds["MEDIUM"]:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_recommendation(self, risk_level: str, score: float) -> str:
        """Generate recommendation based on risk analysis"""
        if risk_level == "CRITICAL":
            return "BLOCK - Order shows critical fraud indicators"
        elif risk_level == "HIGH":
            return "MANUAL_REVIEW - Order requires manual approval"
        elif risk_level == "MEDIUM":
            return "ENHANCED_MONITORING - Allow with increased scrutiny"
        else:
            return "APPROVE - Normal risk profile"


# Global analyzer instance
_analyzer = None


def get_analyzer() -> RiskAnalyzer:
    """Get or create risk analyzer singleton"""
    global _analyzer
    if _analyzer is None:
        _analyzer = RiskAnalyzer()
    return _analyzer
