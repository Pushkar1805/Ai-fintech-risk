from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .db import init_db, get_db
from .models import Customer, Assessment
from .schemas import Application


# ============================================================
# BASE PATH
# ============================================================

BASE = Path(__file__).resolve().parents[1]


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

# Database is configured through backend/.env
# DATABASE_URL should point to your Supabase PostgreSQL database.
init_db()


# ============================================================
# LOAD MACHINE LEARNING MODELS
# ============================================================

credit = joblib.load(
    BASE / "trained_models" / "credit_model.joblib"
)

fraud = joblib.load(
    BASE / "trained_models" / "fraud_model.joblib"
)

anomaly = joblib.load(
    BASE / "trained_models" / "anomaly_model.joblib"
)

FEATURES = credit["features"]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RiskLedger AI API",
    version="2.0.0",
    description=(
        "Alternative credit, behavioral risk and fraud "
        "intelligence API."
    ),
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

# Allows the Next.js frontend to communicate with FastAPI
# during local development on any localhost/127.0.0.1 port.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CREATE MODEL INPUT ROW
# ============================================================

def row(a):
    return pd.DataFrame(
        [
            {
                k: (int(v) if isinstance(v, bool) else v)
                for k, v in a.model_dump().items()
                if k in FEATURES
            }
        ],
        columns=FEATURES,
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

def assess(a):

    # --------------------------------------------------------
    # KYC CHECK
    # --------------------------------------------------------

    if a.kyc_status != "VERIFIED":
        return {
            "kyc_status": a.kyc_status,
            "decision": "MANUAL_REVIEW",
            "risk_level": "HIGH",
            "overall_risk_score": 100,
            "risk_factors": [
                "Digital KYC is not verified."
            ],
        }

    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    x = row(a)

    # --------------------------------------------------------
    # CREDIT MODEL
    # --------------------------------------------------------

    credit_prob = float(
        credit["model"].predict_proba(x)[0, 1]
    )

    alt = int(
        round((1 - credit_prob) * 100)
    )

    credit_risk = 100 - alt

    # --------------------------------------------------------
    # FRAUD MODEL
    # --------------------------------------------------------

    fraud_prob = float(
        fraud["model"].predict_proba(x)[0, 1]
    )

    # --------------------------------------------------------
    # ANOMALY MODEL
    # --------------------------------------------------------

    anomaly_score = float(
        np.clip(
            50
            - anomaly["model"].decision_function(x)[0] * 18,
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # FRAUD RISK
    # --------------------------------------------------------

    fraudrisk = float(
        np.clip(
            0.65 * (fraud_prob * 100)
            + 0.35 * anomaly_score,
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # BEHAVIORAL RISK
    # --------------------------------------------------------

    behavior = float(
        np.clip(
            a.spending_volatility * 55
            + a.failed_payments * 5
            + a.late_payments * 7
            + (
                12
                if a.average_monthly_outflow
                > a.average_monthly_inflow
                else 0
            )
            + (
                10
                if a.account_age_months < 9
                else 0
            ),
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # OVERALL RISK
    # --------------------------------------------------------

    overall = int(
        round(
            np.clip(
                0.5 * credit_risk
                + 0.2 * behavior
                + 0.3 * fraudrisk,
                0,
                100,
            )
        )
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = (
        "APPROVE"
        if overall <= 30 and fraudrisk < 60
        else "REVIEW"
        if overall <= 65 or fraudrisk < 80
        else "DECLINE"
    )

    # --------------------------------------------------------
    # CREDIT LIMIT
    # --------------------------------------------------------

    disposable = max(
        0,
        a.average_monthly_inflow
        - a.average_monthly_outflow,
    )

    base = min(
        a.requested_credit_limit,
        a.monthly_income * 0.45,
        max(
            5000,
            disposable * 0.75,
        ),
    )

    limit = (
        round(
            max(
                0,
                base * (1 - overall / 150),
            ),
            -2,
        )
        if decision == "APPROVE"
        else 0
    )

    # --------------------------------------------------------
    # RISK FACTORS
    # --------------------------------------------------------

    factors = []

    if (
        a.average_monthly_inflow
        >= a.average_monthly_outflow
    ):
        factors.append(
            "Positive monthly cash flow"
        )

    if a.failed_payments <= 1:
        factors.append(
            "Low failed-payment frequency"
        )

    if a.spending_volatility < 0.3:
        factors.append(
            "Stable spending pattern"
        )

    if a.new_device:
        factors.append(
            "New device signal detected"
        )

    if a.failed_logins >= 3:
        factors.append(
            "Repeated failed login attempts"
        )

    if a.location_change:
        factors.append(
            "Location change signal"
        )

    if a.account_age_months < 9:
        factors.append(
            "Limited account history"
        )

    if a.average_transaction > 12000:
        factors.append(
            "High-value transaction pattern"
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "kyc_status": a.kyc_status,

        "alternative_credit_score": alt,

        "behavioral_risk_score": round(
            behavior
        ),

        "fraud_risk_score": round(
            fraudrisk
        ),

        "overall_risk_score": overall,

        "risk_level": (
            "LOW"
            if overall <= 30
            else "MEDIUM"
            if overall <= 65
            else "HIGH"
            if overall <= 80
            else "VERY HIGH"
        ),

        "decision": decision,

        "recommended_credit_limit": int(
            limit
        ),

        "risk_factors": factors[:8],
    }


# ============================================================
# ROOT ROUTE
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "RiskLedger AI API",
        "message": "Backend is running successfully.",
        "docs": "/docs",
        "health": "/health",
        "database": "supabase-postgresql",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "riskledger-ai",
        "database": "supabase-postgresql",
    }


# ============================================================
# DIGITAL KYC
# ============================================================

@app.post("/kyc/verify")
def kyc(payload: dict):

    return {
        "status": "VERIFIED",
        "mode": "Demo Digital KYC",
        "checks": {
            "document": "PASS",
            "selfie": "PASS",
            "liveness": "PASS",
        },
    }


# ============================================================
# APPLICATION ASSESSMENT
# ============================================================

@app.post("/application/assess")
def application(
    a: Application,
    db: Session = Depends(get_db),
):

    # Run AI risk assessment
    r = assess(a)

    # Save assessment to Supabase PostgreSQL
    db.add(
        Assessment(
            customer_id=a.customer_id,

            overall_risk_score=r.get(
                "overall_risk_score",
                100,
            ),

            behavioral_risk_score=r.get(
                "behavioral_risk_score",
                100,
            ),

            fraud_risk_score=r.get(
                "fraud_risk_score",
                100,
            ),

            alternative_credit_score=r.get(
                "alternative_credit_score",
                0,
            ),

            decision=r["decision"],

            recommended_credit_limit=r.get(
                "recommended_credit_limit",
                0,
            ),

            risk_factors=json.dumps(
                r.get(
                    "risk_factors",
                    [],
                )
            ),
        )
    )

    # Save to Supabase
    db.commit()

    return r


# ============================================================
# RISK PREDICTION
# ============================================================

@app.post("/risk/predict")
def risk(a: Application):

    return assess(a)


# ============================================================
# FRAUD CHECK
# ============================================================

@app.post("/fraud/check")
def fraud_check(a: Application):

    r = assess(a)

    return {
        "fraud_risk_score": r.get(
            "fraud_risk_score",
            100,
        ),

        "signals": r.get(
            "risk_factors",
            [],
        ),
    }


# ============================================================
# MODEL METRICS
# ============================================================

@app.get("/model/metrics")
def metrics():

    return json.loads(
        (
            BASE
            / "trained_models"
            / "metrics.json"
        ).read_text()
    )


# ============================================================
# CUSTOMERS
# ============================================================

@app.get("/customers")
def customers(
    limit: int = 20,
    db: Session = Depends(get_db),
):

    rows = (
        db.query(Customer)
        .order_by(Customer.id.desc())
        .limit(limit)
        .all()
    )

    # If Supabase doesn't have customers yet,
    # load them from the CSV and insert them.
    if not rows:

        csv = pd.read_csv(
            BASE / "data" / "customers.csv"
        ).head(limit)

        for _, x in csv.iterrows():

            db.add(
                Customer(
                    customer_id=x.customer_id,
                    monthly_income=float(
                        x.monthly_income
                    ),
                    savings_balance=float(
                        x.savings_balance
                    ),
                )
            )

        db.commit()

        rows = (
            db.query(Customer)
            .order_by(Customer.id.desc())
            .limit(limit)
            .all()
        )

    return [
        {
            "customer_id": x.customer_id,
            "monthly_income": x.monthly_income,
            "savings_balance": x.savings_balance,
            "kyc_status": x.kyc_status,
        }
        for x in rows
    ]


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard")
def dashboard(
    db: Session = Depends(get_db),
):

    return {
        "customers": int(
            pd.read_csv(
                BASE
                / "data"
                / "customers.csv"
            ).shape[0]
        ),

        "assessments": db.query(
            Assessment
        ).count(),

        "model": {
            "credit": credit["name"],
            "fraud": fraud["name"],
        },

        "database": "Supabase PostgreSQL",
    }