from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import sqlite3
import json

conn = sqlite3.connect("db/credit.db", check_same_thread=False)
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS credit_application (
    id_application INTEGER PRIMARY KEY AUTOINCREMENT,
    avg_cur_bal REAL,
    dti REAL,
    acc_open_past_24mths REAL,
    term TEXT,
    inq_last_12m REAL,
    max_bal_bc REAL,
    desc TEXT,
    mths_since_recent_inq REAL,
    open_rv_24m REAL,
    loan_amnt REAL,
    num_actv_rev_tl REAL,
    mort_acc REAL,
    total_bal_il REAL,
    all_util REAL,
    verification_status TEXT,
    home_ownership TEXT,
    int_rate REAL,
    fico_range_low REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS credit_decision (
    id_decision INTEGER PRIMARY KEY AUTOINCREMENT,
    id_application INTEGER,
    pd REAL,
    risk_grade TEXT,
    lgd REAL,
    expected_loss REAL,
    profitability REAL,
    decision TEXT,
    reason_codes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_application) REFERENCES credit_application(id_application)
)
""")

conn.commit()

model = joblib.load("calib_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
top_features = joblib.load("top_features.pkl")

app = FastAPI()

# Request schema 
class LoanFeatures(BaseModel):
    avg_cur_bal: float = 5000
    dti: float = 0.25
    acc_open_past_24mths: float = 2
    term: str = "60 months"
    inq_last_12m: float = 1
    max_bal_bc: float = 10000
    desc: str = "Debt consolidation"
    mths_since_recent_inq: float = 6
    open_rv_24m: float = 3
    loan_amnt: float = 15000
    num_actv_rev_tl: float = 4
    mort_acc: float = 1
    total_bal_il: float = 8000
    all_util: float = 0.45
    verification_status: str = "Source Verified"
    home_ownership: str = "RENT"
    int_rate: float = 12.5
    fico_range_low: float = 700
    sec_app_open_act_il: float = 0
    total_rev_hi_lim: float = 0
    mths_since_recent_il: float = 0
    il_util: float = 0
    sec_app_revol_util: float = 0
    total_bal_ex_mort: float = 0
    sec_app_earliest_cr_line: str = "2010-01"
    open_acc: float = 0
    num_bc_tl: float = 0
    annual_inc: float = 50000
    addr_state: str = "CA"
    sec_app_fico_range_low: float = 0
    pub_rec_bankruptcies: float = 0
    open_act_il: float = 0
    mths_since_last_major_derog: float = 0
    mths_since_last_record: float = 0
    num_tl_30dpd: float = 0
    revol_bal: float = 0
    total_cu_tl: float = 0
    initial_list_status: str = "W"
    bc_util: float = 0
    sec_app_mort_acc: float = 0
    acc_now_delinq: float = 0
    sec_app_collections_12_mths_ex_med: float = 0
    title: str = "Other"
    mths_since_recent_revol_delinq: float = 0
    purpose: str = "debt_consolidation"
    mths_since_recent_bc: float = 0
    tax_liens: float = 0
    mths_since_recent_bc_dlq: float = 0
    revol_util: float = 0.45
    verification_status_joint: str = "Not Verified"
    sec_app_open_acc: float = 0
    sec_app_num_rev_accts: float = 0
    collections_12_mths_ex_med: float = 0
    application_type: str = "Individual"
    open_acc_6m: float = 0
    pct_tl_nvr_dlq: float = 100
    tot_coll_amt: float = 0
    num_il_tl: float = 0
    percent_bc_gt_75: float = 0
    annual_inc_joint: float = 0
    num_actv_bc_tl: float = 0
    num_op_rev_tl: float = 0
    num_tl_90g_dpd_24m: float = 0
    inq_fi: float = 0
    delinq_amnt: float = 0
    mths_since_last_delinq: float = 0
    chargeoff_within_12_mths: float = 0
    sec_app_inq_last_6mths: float = 0
    sec_app_mths_since_last_major_derog: float = 0
    mo_sin_old_il_acct: float = 0
    open_il_24m: float = 0
    bc_open_to_buy: float = 0
    tot_cur_bal: float = 0
    open_il_12m: float = 0
    earliest_cr_line: str = "2010-01"
    disbursement_method: str = "Cash"
    num_bc_sats: float = 0
    num_rev_accts: float = 0
    total_il_high_credit_limit: float = 0
    total_bc_limit: float = 0
    open_rv_12m: float = 0
    num_accts_ever_120_pd: float = 0
    revol_bal_joint: float = 0
    inq_last_6mths: float = 0
    pub_rec: float = 0
    emp_length: str = "10+ years"
    num_tl_op_past_12m: float = 0
    delinq_2yrs: float = 0
    total_acc: float = 10
    mo_sin_rcnt_tl: float = 0
    mo_sin_rcnt_rev_tl_op: float = 0
    dti_joint: float = 0
    mths_since_rcnt_il: float = 0
    mo_sin_old_rev_tl_op: float = 0

def map_pd_to_risk_grade(pd: float) -> str:
    if pd < 0.01:
        return "AAA"
    elif pd < 0.03:
        return "AA"
    elif pd < 0.06:
        return "A"
    elif pd < 0.10:
        return "BBB"
    elif pd < 0.20:
        return "BB"
    elif pd < 0.40:
        return "B"
    else:
        return "CCC"
def eligibility_rules(features: LoanFeatures):
    reasons = []

    
    
    MIN_AGE = 21
    MAX_AGE = 65

    # (Temporary proxy: earliest_cr_line → rough age safeguard)
    if features.annual_inc < 12000:
        reasons.append("Income below minimum threshold")

    #  DTI rule
    if features.dti > 0.45:
        reasons.append("DTI exceeds maximum allowed")

    # Loan to income rule
    if features.loan_amnt > 10 * features.annual_inc:
        reasons.append("Loan amount too high relative to income")

    # Final eligibility decision
    return {
        "eligible": len(reasons) == 0,
        "rejection_reasons": reasons
    }

@app.post("/predict")
def predict(features: LoanFeatures):

    # -----------------------------
    # Step 1: Eligibility check
    # -----------------------------
    eligibility = eligibility_rules(features)
    reason_codes = []

    if not eligibility["eligible"]:
        reason_codes.extend(eligibility["rejection_reasons"])
        return {
            "decision": "REJECT",
            "stage": "ELIGIBILITY",
            "reasons": reason_codes
        }

    # -----------------------------
    # Step 2: Prepare data & compute PD
    # -----------------------------
    df = pd.DataFrame([features.dict()])

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
        else:
            df[col] = df[col].astype(float)

    X_enc = preprocessor.transform(df)
    X_enc = pd.DataFrame(X_enc, columns=preprocessor.get_feature_names_out())
    X_model = X_enc[top_features]

    pd_proba = model.predict_proba(X_model)[0, 1]
    risk_grade = map_pd_to_risk_grade(pd_proba)

    # -----------------------------
    # Step 3: Risk metrics
    # -----------------------------
    product_lgd_map = {
        "debt_consolidation": 0.45,
        "auto": 0.30,
        "mortgage": 0.20
    }

    lgd = product_lgd_map.get(features.purpose, 0.45)
    ead = features.loan_amnt
    expected_loss_value = round(pd_proba * lgd * ead, 2)

    interest_income = ead * (features.int_rate / 100)
    profit = round(interest_income - expected_loss_value, 2)

    # -----------------------------
    # Step 4: Decision logic
    # -----------------------------
    decision = "APPROVE"

    if pd_proba > 0.40:
        decision = "REJECT"
        reason_codes.append("PD too high")
    elif profit <= 0:
        decision = "REJECT"
        reason_codes.append("Expected loss exceeds interest income")
    elif pd_proba > 0.20:
        decision = "APPROVE_CONDITIONAL"
        reason_codes.append("High risk – conditional approval")

    # -----------------------------
    # Step 5: Store loan application
    # -----------------------------
    cursor.execute("""
        INSERT INTO credit_application (
            avg_cur_bal, dti, acc_open_past_24mths, term,
            inq_last_12m, max_bal_bc, desc, mths_since_recent_inq,
            open_rv_24m, loan_amnt, num_actv_rev_tl, mort_acc,
            total_bal_il, all_util, verification_status,
            home_ownership, int_rate, fico_range_low
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        features.avg_cur_bal,
        features.dti,
        features.acc_open_past_24mths,
        features.term,
        features.inq_last_12m,
        features.max_bal_bc,
        features.desc,
        features.mths_since_recent_inq,
        features.open_rv_24m,
        features.loan_amnt,
        features.num_actv_rev_tl,
        features.mort_acc,
        features.total_bal_il,
        features.all_util,
        features.verification_status,
        features.home_ownership,
        features.int_rate,
        features.fico_range_low
    ))

    id_application = cursor.lastrowid

    # -----------------------------
    # Step 6: Store credit decision
    # -----------------------------
    cursor.execute("""
        INSERT INTO credit_decision (
            id_application, pd, risk_grade, lgd,
            expected_loss, profitability, decision, reason_codes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        id_application,
        round(pd_proba, 4),
        risk_grade,
        lgd,
        expected_loss_value,
        profit,
        decision,
        json.dumps(reason_codes)
    ))

    conn.commit()

    # -----------------------------
    # Step 7: API response
    # -----------------------------
    return {
        "decision": decision,
        "stage": "FULL_DECISION",
        "pd": round(pd_proba, 4),
        "risk_grade": risk_grade,
        "lgd": lgd,
        "expected_loss": expected_loss_value,
        "profitability": profit,
        "reason_codes": reason_codes
    }
