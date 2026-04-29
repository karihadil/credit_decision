"""
Credit Decision API
===================
FastAPI service exposing the calibrated XGBoost credit scoring model.

IMPORTANT: The preprocessing in run_decision_logic() MUST match
what train_model.py does during training. If you change one, change both.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration — centralise all business constants here
# ---------------------------------------------------------------------------
DB_PATH = "db/credit.db"

# Loss Given Default by loan purpose (source: internal backtesting)
LGD_BY_PURPOSE: dict[str, float] = {
    "debt_consolidation": 0.45,
    "auto": 0.30,
    "mortgage": 0.20,
}
LGD_DEFAULT = 0.45  # Fallback for unlisted purposes

# PD decision thresholds
PD_REJECT_THRESHOLD      = 0.40   # Hard reject above this
PD_CONDITIONAL_THRESHOLD = 0.20   # Conditional approval above this

# Reference date for credit history calculation (same approach as training)
REFERENCE_DATE = pd.Timestamp.now()

# ---------------------------------------------------------------------------
# Database — WAL mode + per-request connections + write lock
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()


def _init_db() -> None:
    """Create tables on startup (idempotent)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS credit_application (
            id_application        INTEGER PRIMARY KEY AUTOINCREMENT,
            avg_cur_bal           REAL,
            dti                   REAL,
            acc_open_past_24mths  REAL,
            term                  TEXT,
            inq_last_12m          REAL,
            max_bal_bc            REAL,
            mths_since_recent_inq REAL,
            open_rv_24m           REAL,
            loan_amnt             REAL,
            num_actv_rev_tl       REAL,
            mort_acc              REAL,
            total_bal_il          REAL,
            all_util              REAL,
            verification_status   TEXT,
            annual_inc            REAL,
            home_ownership        TEXT,
            int_rate              REAL,
            fico_range_low        REAL,
            created_at            TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS credit_decision (
            id_decision    INTEGER PRIMARY KEY AUTOINCREMENT,
            id_application INTEGER,
            pd             REAL,
            risk_grade     TEXT,
            lgd            REAL,
            expected_loss  REAL,
            profitability  REAL,
            decision       TEXT,
            reason_codes   TEXT,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_application) REFERENCES credit_application(id_application)
        );
    """)
    conn.commit()
    conn.close()


_init_db()


@contextmanager
def get_db():
    """Yield a fresh per-request SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Model loading — fail fast with a readable message at startup
# ---------------------------------------------------------------------------
try:
    model        = joblib.load("calib_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    top_features = joblib.load("top_features.pkl")
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Model artifact not found: {exc}. Run train_model.py first."
    ) from exc

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Decision API",
    description="Credit scoring and decision engine based on a calibrated XGBoost PD model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # TODO: restrict to your frontend domain in production
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request schema — with field-level validators
# ---------------------------------------------------------------------------
class LoanFeatures(BaseModel):
    # Key fields with value constraints
    loan_amnt:            float = Field(default=15000, gt=0,          description="Loan amount (USD)")
    annual_inc:           float = Field(default=50000, gt=0,          description="Annual income (USD)")
    int_rate:             float = Field(default=12.5,  gt=0, le=100,  description="Interest rate (%)")
    dti:                  float = Field(default=18.0,  ge=0, le=999,  description="Debt-to-income ratio (%)")
    fico_range_low:       float = Field(default=700,   ge=300, le=850)
    avg_cur_bal:          float = Field(default=5000,  ge=0)
    max_bal_bc:           float = Field(default=10000, ge=0)
    total_bal_il:         float = Field(default=8000,  ge=0)
    all_util:             float = Field(default=45.0,  ge=0, le=200,  description="All accounts utilization (%%, 0-100+ scale)")
    inq_last_12m:         float = Field(default=1,     ge=0)
    acc_open_past_24mths: float = Field(default=2,     ge=0)
    open_rv_24m:          float = Field(default=3,     ge=0)
    num_actv_rev_tl:      float = Field(default=4,     ge=0)
    mort_acc:             float = Field(default=1,     ge=0)
    mths_since_recent_inq: float = Field(default=6,   ge=0)

    # Categorical fields
    term:                     str = " 60 months"       # note: leading space matches training data
    verification_status:      str = "Source Verified"
    home_ownership:           str = "RENT"
    purpose:                  str = "debt_consolidation"
    initial_list_status:      str = "W"                # uppercase to match training
    application_type:         str = "Individual"
    disbursement_method:      str = "Cash"
    emp_length:               str = "10+ years"
    addr_state:               str = "CA"
    verification_status_joint: str = "Not Verified"

    # Date fields — will be converted to numeric months at inference
    earliest_cr_line:         str = "Jan-2010"
    sec_app_earliest_cr_line: str = "Jan-2010"

    # Supplemental / secondary fields (default None -> np.nan)
    sec_app_open_act_il:                float | None = None
    total_rev_hi_lim:                   float | None = None
    mths_since_recent_il:               float | None = None
    il_util:                            float | None = None
    sec_app_revol_util:                 float | None = None
    total_bal_ex_mort:                  float | None = None
    open_acc:                           float | None = None
    num_bc_tl:                          float | None = None
    sec_app_fico_range_low:             float | None = None
    pub_rec_bankruptcies:               float | None = None
    open_act_il:                        float | None = None
    mths_since_last_major_derog:        float | None = None
    mths_since_last_record:             float | None = None
    num_tl_30dpd:                       float | None = None
    revol_bal:                          float | None = None
    total_cu_tl:                        float | None = None
    bc_util:                            float | None = None
    sec_app_mort_acc:                   float | None = None
    acc_now_delinq:                     float | None = None
    sec_app_collections_12_mths_ex_med: float | None = None
    mths_since_recent_revol_delinq:     float | None = None
    mths_since_recent_bc:               float | None = None
    tax_liens:                          float | None = None
    mths_since_recent_bc_dlq:           float | None = None
    revol_util:                         float = 45.0
    sec_app_open_acc:                   float | None = None
    sec_app_num_rev_accts:              float | None = None
    collections_12_mths_ex_med:         float | None = None
    open_acc_6m:                        float | None = None
    pct_tl_nvr_dlq:                     float = 100
    tot_coll_amt:                       float | None = None
    num_il_tl:                          float | None = None
    percent_bc_gt_75:                   float | None = None
    annual_inc_joint:                   float | None = None
    num_actv_bc_tl:                     float | None = None
    num_op_rev_tl:                      float | None = None
    num_tl_90g_dpd_24m:                 float | None = None
    inq_fi:                             float | None = None
    delinq_amnt:                        float | None = None
    mths_since_last_delinq:             float | None = None
    chargeoff_within_12_mths:           float | None = None
    sec_app_inq_last_6mths:             float | None = None
    sec_app_mths_since_last_major_derog: float | None = None
    mo_sin_old_il_acct:                 float | None = None
    open_il_24m:                        float | None = None
    bc_open_to_buy:                     float | None = None
    tot_cur_bal:                        float | None = None
    open_il_12m:                        float | None = None
    num_bc_sats:                        float | None = None
    num_rev_accts:                      float | None = None
    total_il_high_credit_limit:         float | None = None
    total_bc_limit:                     float | None = None
    open_rv_12m:                        float | None = None
    num_accts_ever_120_pd:              float | None = None
    revol_bal_joint:                    float | None = None
    inq_last_6mths:                     float | None = None
    pub_rec:                            float | None = None
    num_tl_op_past_12m:                 float | None = None
    delinq_2yrs:                        float | None = None
    total_acc:                          float = 10
    mo_sin_rcnt_tl:                     float | None = None
    mo_sin_rcnt_rev_tl_op:              float | None = None
    dti_joint:                          float | None = None
    mths_since_rcnt_il:                 float | None = None
    mo_sin_old_rev_tl_op:               float | None = None

# ---------------------------------------------------------------------------
# Feature engineering helpers — MUST match train_model.py exactly
# ---------------------------------------------------------------------------
def _date_str_to_months(val: str) -> float:
    """Convert 'Mon-YYYY' or 'YYYY-MM' to months of credit history."""
    if pd.isna(val) or str(val).strip() == "":
        return 0.0
    for fmt in ("%Y-%m", "%b-%Y", "%B-%Y"):
        try:
            dt = datetime.strptime(str(val).strip(), fmt)
            return float((REFERENCE_DATE.year - dt.year) * 12 + (REFERENCE_DATE.month - dt.month))
        except ValueError:
            continue
    return 0.0


def _prepare_features(features: LoanFeatures) -> pd.DataFrame:
    """
    Transform API input into the exact DataFrame the preprocessor expects.
    This MUST mirror the feature engineering in train_model.py sections 5-6.
    """
    data = features.model_dump()

    # Convert None values to np.nan so the SimpleImputer kicks in
    for k, v in data.items():
        if v is None:
            data[k] = float('nan')

    data["credit_history_months"] = _date_str_to_months(data.pop("earliest_cr_line"))
    data["sec_app_credit_history_months"] = _date_str_to_months(data.pop("sec_app_earliest_cr_line"))

    data["initial_list_status"] = data["initial_list_status"].upper()

    data["loan_to_income"]   = data["loan_amnt"] / (data["annual_inc"] + 1)
    data["monthly_debt_est"] = (data["dti"] / 100) * data["annual_inc"] / 12
    data["fico_dti_score"]   = data["fico_range_low"] * (1 - data["dti"] / 100)

    df = pd.DataFrame([data])
    mths_cols = [c for c in df.columns
                 if "mths_since" in c or "mo_sin" in c or "months_since" in c]
    for col in mths_cols:
        df[col] = df[col].replace(0, 999)

    df["term"] = df["term"].astype(str).str.strip()
    df["term"] = df["term"].apply(lambda x: x if x.startswith(" ") else " " + x)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    return df


# ---------------------------------------------------------------------------
# DB helpers — single source of truth for inserts
# ---------------------------------------------------------------------------
def _save_application(conn: sqlite3.Connection, f: LoanFeatures) -> int:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO credit_application (
            avg_cur_bal, dti, acc_open_past_24mths, term,
            inq_last_12m, max_bal_bc, mths_since_recent_inq,
            open_rv_24m, loan_amnt, num_actv_rev_tl, mort_acc,
            total_bal_il, all_util, verification_status, annual_inc,
            home_ownership, int_rate, fico_range_low
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        f.avg_cur_bal, f.dti, f.acc_open_past_24mths, f.term,
        f.inq_last_12m, f.max_bal_bc, f.mths_since_recent_inq,
        f.open_rv_24m, f.loan_amnt, f.num_actv_rev_tl, f.mort_acc,
        f.total_bal_il, f.all_util, f.verification_status, f.annual_inc,
        f.home_ownership, f.int_rate, f.fico_range_low,
    ))
    return cur.lastrowid


def _save_decision(conn: sqlite3.Connection, app_id: int, d: dict) -> None:
    conn.execute("""
        INSERT INTO credit_decision (
            id_application, pd, risk_grade, lgd,
            expected_loss, profitability, decision, reason_codes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        app_id, d["pd"], d["risk_grade"], d["lgd"],
        d["expected_loss"], d["profitability"], d["decision"],
        json.dumps(d["reason_codes"]),
    ))

# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------
def map_pd_to_risk_grade(pd_val: float) -> str:
    if pd_val < 0.01: return "AAA"
    if pd_val < 0.03: return "AA"
    if pd_val < 0.06: return "A"
    if pd_val < 0.10: return "BBB"
    if pd_val < 0.20: return "BB"
    if pd_val < 0.40: return "B"
    return "CCC"


def eligibility_rules(f: LoanFeatures) -> dict:
    reasons = []
    if f.annual_inc < 12000:
        reasons.append("Income below minimum threshold ($12,000)")
    if f.dti > 45:
        reasons.append("DTI exceeds maximum allowed (45%)")
    if f.loan_amnt > 0.5 * f.annual_inc:
        reasons.append("Loan amount > 50% of annual income")
    return {"eligible": len(reasons) == 0, "rejection_reasons": reasons}


def run_decision_logic(features: LoanFeatures) -> dict:
    eligibility  = eligibility_rules(features)
    reason_codes: list = []
    soft_reject  = False

    if not eligibility["eligible"]:
        reason_codes.extend(eligibility["rejection_reasons"])
        soft_reject = True

    df = _prepare_features(features)

    X_enc   = preprocessor.transform(df)
    X_enc   = pd.DataFrame(X_enc, columns=preprocessor.get_feature_names_out())
    X_model = X_enc[top_features]

    pd_proba   = float(model.predict_proba(X_model)[0, 1])
    risk_grade = map_pd_to_risk_grade(pd_proba)

    # EL = PD × LGD × EAD
    lgd           = LGD_BY_PURPOSE.get(features.purpose, LGD_DEFAULT)
    ead           = features.loan_amnt
    expected_loss = round(pd_proba * lgd * ead, 2)
    profit        = round(ead * (features.int_rate / 100) - expected_loss, 2)

    # Decision waterfall
    if soft_reject:
        decision = "REJECT"
        reason_codes.append("Failed eligibility rules")
    elif pd_proba > PD_REJECT_THRESHOLD:
        decision = "REJECT"
        reason_codes.append(f"PD too high ({pd_proba:.1%} > {PD_REJECT_THRESHOLD:.0%})")
    elif profit <= 0:
        decision = "APPROVE_CONDITIONAL"
        reason_codes.append("Low expected profitability")
    elif pd_proba > PD_CONDITIONAL_THRESHOLD:
        decision = "APPROVE_CONDITIONAL"
        reason_codes.append(f"Elevated risk ({pd_proba:.1%} > {PD_CONDITIONAL_THRESHOLD:.0%})")
    else:
        decision = "APPROVE"

    return {
        "pd":            round(pd_proba, 4),
        "risk_grade":    risk_grade,
        "lgd":           lgd,
        "expected_loss": expected_loss,
        "profitability": profit,
        "decision":      decision,
        "reason_codes":  reason_codes,
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Ops"])
def health():
    return {
        "status":   "ok",
        "model":    "calib_model.pkl",
        "features": len(top_features),
    }


@app.post("/predict", tags=["Scoring"])
def predict(features: LoanFeatures):
    decision_data = run_decision_logic(features)
    with _write_lock, get_db() as conn:
        app_id = _save_application(conn, features)
        _save_decision(conn, app_id, decision_data)
        conn.commit()
    return {"id_application": app_id, "stage": "FULL DECISION", **decision_data}


@app.post("/predict/bulk", tags=["Scoring"])
def predict_bulk(applications: List[LoanFeatures]):
    results = []
    with _write_lock, get_db() as conn:
        for features in applications:
            decision_data = run_decision_logic(features)
            app_id        = _save_application(conn, features)
            _save_decision(conn, app_id, decision_data)
            results.append({"id_application": app_id, **decision_data})
        conn.commit()  # Single atomic commit for the whole batch
    return {"count": len(results), "results": results}


@app.get("/applications", tags=["Data"])
def get_applications(limit: int = 10, offset: int = 0):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT
                app.id_application, app.loan_amnt, app.created_at,
                dec.decision, dec.risk_grade, dec.pd, dec.reason_codes
            FROM credit_application app
            LEFT JOIN credit_decision dec
                   ON app.id_application = dec.id_application
            ORDER BY app.created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

        total = conn.execute(
            "SELECT COUNT(*) FROM credit_application"
        ).fetchone()[0]

    applications = []
    for row in rows:
        try:
            reason_codes = json.loads(row[6]) if row[6] else []
        except (json.JSONDecodeError, TypeError):
            reason_codes = []

        applications.append({
            "id_application": row[0],
            "loan_amnt":      row[1],
            "created_at":     row[2],
            "decision":       row[3] or "PENDING",
            "risk_grade":     row[4] or "N/A",
            "pd":             row[5] or 0.0,
            "reason_codes":   reason_codes,
            "client_name":    f"Applicant #{row[0]}",
        })

    return {
        "data":   applications,
        "total":  total,
        "limit":  limit,
        "offset": offset,
    }
