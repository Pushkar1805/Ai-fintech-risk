# RiskLedger AI — Behavioral & Fraud Risk Intelligence

A portfolio-quality fintech prototype for alternative credit assessment after Digital/Virtual KYC. It does **not** use CIBIL or traditional bureau scores. It uses synthetic data only.

## Architecture

Next.js UI → FastAPI → ML risk engine → SQLite database

### AI models
- **Alternative credit risk:** Logistic Regression, Random Forest and XGBoost are trained and compared; the best ROC-AUC model is persisted.
- **Fraud risk:** supervised fraud classifier + Isolation Forest anomaly score.
- **Behavioral risk:** interpretable behavioral signal engine using spending volatility, payment behavior, cash flow and account history.
- **Decision engine:** APPROVE / REVIEW / DECLINE plus recommended credit limit and human-readable reason codes.

## No Docker

Docker is intentionally removed. Run the services directly on Windows, macOS or Linux.

### 1. Backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python ml/bootstrap.py
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs
Database: `backend/risk.db` (SQLite, created automatically)

### 2. Frontend
Open another terminal:
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:3000

## Tests
```bash
cd backend
pytest -q
```

## Important
This is a research/demo underwriting system. Digital KYC is simulated, training labels are synthetic, and real lending/card issuance requires applicable KYC, AML, fair-lending, model-risk and regulatory controls.
