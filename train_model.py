import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    auc, average_precision_score, classification_report,
    f1_score, precision_recall_curve,
    recall_score, roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
REFERENCE_DATE = pd.Timestamp.now()

print("Loading data (chunked to save memory)...")
# ── FIX: Read in chunks to handle 2.3 GB file without OOM
chunks = []
for chunk in pd.read_csv("accepted_imputed.csv", chunksize=200_000, low_memory=False):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
del chunks

# Downcast numeric columns to save ~50% memory
for col in df.select_dtypes(include="float64").columns:
    df[col] = pd.to_numeric(df[col], downcast="float")
for col in df.select_dtypes(include="int64").columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")

print(f"   Raw shape: {df.shape}")
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print(df["loan_status"].value_counts())

DEFAULT_STATUSES = {
    "Charged Off", "Default", "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
}
PAID_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

def map_status(s: str):
    if s in DEFAULT_STATUSES: return 1
    if s in PAID_STATUSES:    return 0
    return None   # "Current", "In Grace Period", "Late (16-30 days)" → drop

df["default_flag"] = df["loan_status"].apply(map_status)
df = df.dropna(subset=["default_flag"])
df["default_flag"] = df["default_flag"].astype(int)
print(f"\n   Target: {df['default_flag'].value_counts().to_dict()}")

def drop_id_columns(df):
    cols = [c for c in df.columns if "id" in c.lower()]
    return df.drop(columns=cols, errors="ignore"), cols

def drop_constant_features(df, threshold=0.999):
    cols = [c for c in df.columns
            if df[c].value_counts(normalize=True, dropna=False).values[0] >= threshold]
    return df.drop(columns=cols, errors="ignore"), cols

def drop_high_cardinality(df, max_ratio=0.2):
    cols = [c for c in df.select_dtypes("object").columns
            if df[c].nunique() / len(df) > max_ratio]
    return df.drop(columns=cols, errors="ignore"), cols

def drop_correlated(df, threshold=0.95):
    mat   = df.corr(numeric_only=True).abs()
    upper = mat.where(np.triu(np.ones(mat.shape), k=1).astype(bool))
    cols  = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(columns=cols, errors="ignore"), cols

def detect_leakage(df):
    keywords = ("settlement", "recoveries", "default")
    return [c for c in df.columns if any(k in c.lower() for k in keywords)]

def feature_audit(df, target_col="default_flag"):
    dropped = {}
    df, dropped["id_like"]   = drop_id_columns(df)
    df, dropped["constant"]  = drop_constant_features(df)
    df, dropped["high_card"] = drop_high_cardinality(df)
    df, dropped["correlated"]= drop_correlated(df)
    dropped["leakage"]       = [c for c in detect_leakage(df) if c != target_col]
    return df, dropped

clean_df, dropped_info = feature_audit(df, target_col="default_flag")
clean_df = clean_df.drop(columns=dropped_info["leakage"], errors="ignore")

print("\nAutomated audit dropped:")
for k, v in dropped_info.items():
    print(f"   {k}: {len(v)} -> {v[:5]}{'...' if len(v) > 5 else ''}")

POST_LOAN_COLS = [
    "int_rate",  # ← lender-assigned rate = circular leakage for a PD model
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
    "policy_code", "emp_title", "zip_code", "issue_d",
    "hardship_reason", "hardship_status", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date",
    "grade", "sub_grade",               
    "hardship_amount", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    "hardship_payoff_balance_amount",
    "orig_projected_additional_accrued_interest",
    "loan_status",
]
clean_df = clean_df.drop(columns=POST_LOAN_COLS, errors="ignore")
clean_df = clean_df.drop_duplicates()
print(f"\n   Shape after leakage drops: {clean_df.shape}")

print("\nEngineering features...")

def date_str_to_months(series: pd.Series) -> pd.Series:
    """'Mon-YYYY' or 'YYYY-MM' → months of history from today."""
    def _convert(val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        for fmt in ("%Y-%m", "%b-%Y", "%B-%Y"):
            try:
                dt = datetime.strptime(str(val).strip(), fmt)
                return (REFERENCE_DATE.year - dt.year) * 12 + (REFERENCE_DATE.month - dt.month)
            except ValueError:
                continue
        return np.nan
    return series.apply(_convert)

if "earliest_cr_line" in clean_df.columns:
    clean_df["credit_history_months"] = date_str_to_months(clean_df["earliest_cr_line"])
    clean_df = clean_df.drop(columns=["earliest_cr_line"])

if "sec_app_earliest_cr_line" in clean_df.columns:
    clean_df["sec_app_credit_history_months"] = date_str_to_months(clean_df["sec_app_earliest_cr_line"])
    clean_df = clean_df.drop(columns=["sec_app_earliest_cr_line"])

clean_df = clean_df.drop(columns=["desc"], errors="ignore")

clean_df = clean_df.drop(columns=["title"], errors="ignore")


if "initial_list_status" in clean_df.columns:
    clean_df["initial_list_status"] = clean_df["initial_list_status"].str.upper()

clean_df["loan_to_income"]   = clean_df["loan_amnt"] / (clean_df["annual_inc"] + 1)
clean_df["monthly_debt_est"] = (clean_df["dti"] / 100) * clean_df["annual_inc"] / 12
clean_df["fico_dti_score"]   = clean_df["fico_range_low"] * (1 - clean_df["dti"] / 100)

print(f"   Shape after engineering: {clean_df.shape}")
print("\nImputing missing values...")

X_raw = clean_df.drop(columns=["default_flag"])
y     = clean_df["default_flag"].astype(int)

mths_cols = [c for c in X_raw.columns
             if "mths_since" in c or "mo_sin" in c or "months_since" in c]
X_raw[mths_cols] = X_raw[mths_cols].fillna(999)


print(f"   Final shape — X: {X_raw.shape}  y: {y.shape}")


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_valthresh, y_tr, y_valthresh = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)
X_val, X_thresh, y_val, y_thresh = train_test_split(
    X_valthresh, y_valthresh, test_size=0.35, random_state=42, stratify=y_valthresh)

print(f"\nSplit  Train={len(X_tr)}  Val={len(X_val)}  Thresh={len(X_thresh)}  Test={len(X_test)}")
print(f"   Default rate — train: {y_tr.mean():.2%}  val: {y_val.mean():.2%}  "
      f"thresh: {y_thresh.mean():.2%}  test: {y_test.mean():.2%}")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

cat_features = X_raw.select_dtypes(include="object").columns.tolist()
num_features = X_raw.select_dtypes(include="number").columns.tolist()

print(f"\n   Numeric features to impute (Median): {len(num_features)}")
print(f"   Categorical features to impute/OHE:  {len(cat_features)}")

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

num_transformer = SimpleImputer(strategy="median")

transformers = [
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features),
]

preprocessor = ColumnTransformer(transformers, remainder="passthrough")

X_tr_enc    = preprocessor.fit_transform(X_tr)
X_val_enc   = preprocessor.transform(X_val)
X_thresh_enc = preprocessor.transform(X_thresh)
X_test_enc  = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

feature_names = [
    n.replace("[", "(").replace("]", ")").replace("<", "_lt_")
    for n in feature_names
]
X_tr_enc    = pd.DataFrame(X_tr_enc,    columns=feature_names)
X_val_enc   = pd.DataFrame(X_val_enc,   columns=feature_names)
X_thresh_enc = pd.DataFrame(X_thresh_enc, columns=feature_names)
X_test_enc  = pd.DataFrame(X_test_enc,  columns=feature_names)


print("\nFeature selection pass...")
scale_pos_weight = y_tr.value_counts()[0] / y_tr.value_counts()[1]
print(f"   Class ratio (neg/pos): {scale_pos_weight:.2f}")

xgb_selector = XGBClassifier(
    learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight, n_estimators=1000,
    objective="binary:logistic", eval_metric="auc", random_state=42, n_jobs=-1,
    early_stopping_rounds=30,   # XGBoost 3.x: must be in constructor, not fit()
)
xgb_selector.fit(
    X_tr_enc, y_tr,
    eval_set=[(X_val_enc, y_val)],
    verbose=False,
)

feat_importances = pd.Series(xgb_selector.feature_importances_, index=feature_names)
top_features     = feat_importances.sort_values(ascending=False).head(25).index.tolist()
joblib.dump(top_features, "top_features.pkl", protocol=4)

X_tr_top     = X_tr_enc[top_features]
X_val_top    = X_val_enc[top_features]
X_thresh_top = X_thresh_enc[top_features]
X_test_top   = X_test_enc[top_features]
print(f"   Top features: {top_features}")

print("\nTraining final calibrated model...")

xgb_model = XGBClassifier(
    learning_rate=0.0453,
    max_depth=3,
    subsample=0.723,
    colsample_bytree=0.959,
    min_child_weight=1,
    gamma=3.574,
    scale_pos_weight=scale_pos_weight,  
    n_estimators=1000,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

X_calib = pd.concat([X_tr_top, X_val_top])
y_calib = pd.concat([y_tr, y_val])

calib = CalibratedClassifierCV(xgb_model, method="isotonic", cv=5)
calib.fit(X_calib, y_calib)


y_thresh_proba = calib.predict_proba(X_thresh_top)[:, 1]
thresholds  = np.arange(0.10, 0.55, 0.01)
f1_scores   = [f1_score(y_thresh, (y_thresh_proba > t).astype(int)) for t in thresholds]
best_t      = float(thresholds[np.argmax(f1_scores)])
print(f"\n   Optimal threshold (thresh-set F1-max): {best_t:.2f}")

y_test_proba = calib.predict_proba(X_test_top)[:, 1]
y_pred       = (y_test_proba > best_t).astype(int)

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc     = auc(fpr, tpr)
ks          = float(np.max(tpr - fpr))

print("\n" + "="*55)
print("TEST RESULTS")
print("="*55)
print(f"  ROC AUC:           {roc_auc:.4f}")
print(f"  KS Statistic:      {ks:.4f}")
print(f"  F1 (t={best_t:.2f}):       {f1_score(y_test, y_pred):.4f}")
print(f"  Recall:            {recall_score(y_test, y_pred):.4f}")
print(f"  Avg Precision:     {average_precision_score(y_test, y_test_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("  Running 5-fold CV (AUC)...")
cv_scores = cross_val_score(
    XGBClassifier(**{k: v for k, v in xgb_model.get_params().items()}),
    X_calib, y_calib, cv=5, scoring="roc_auc", n_jobs=-1,
)
print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Decile test
def decile_test(y_true, y_proba, n_bins=10):
    d = pd.DataFrame({"y_true": y_true.values, "y_proba": y_proba})
    d["decile"] = pd.qcut(d["y_proba"], q=n_bins, labels=False, duplicates="drop")
    return d.groupby("decile").agg(
        predicted_pd=("y_proba", "mean"),
        observed_default=("y_true", "mean"),
        count=("y_true", "size"),
    ).reset_index().sort_values("decile")

print("\nDecile Test:\n", decile_test(y_test, y_test_proba))

fitted_xgb  = calib.calibrated_classifiers_[0].estimator
feat_imp_df = pd.Series(fitted_xgb.feature_importances_, index=X_tr_top.columns)
top20       = feat_imp_df.sort_values(ascending=False).head(20)

prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
ap           = average_precision_score(y_test, y_test_proba)
prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("XGBoost Credit Default Model — Evaluation", fontsize=14, fontweight="bold")

sns.barplot(x=top20.values, y=top20.index, palette="viridis", ax=axes[0, 0])
axes[0, 0].set_title("Top 20 Feature Importances")
axes[0, 0].set_xlabel("Importance Score")

axes[0, 1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
axes[0, 1].plot([0, 1], [0, 1], "r--")
axes[0, 1].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
axes[0, 1].legend()

axes[1, 0].plot(rec, prec, color="green", lw=2, label=f"AP = {ap:.3f}")
axes[1, 0].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
axes[1, 0].legend()

axes[1, 1].plot(prob_pred, prob_true, "o-", label="Calibrated model")
axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
axes[1, 1].set(title="Calibration Curve (PD vs Observed)", xlabel="Predicted PD", ylabel="Observed default rate")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
print("   Plots saved to model_evaluation.png")


joblib.dump(calib,        "calib_model.pkl",   protocol=4)
joblib.dump(preprocessor, "preprocessor.pkl",  protocol=4)


print("\nArtifacts saved: calib_model.pkl, preprocessor.pkl, top_features.pkl")
print(f"   Optimal inference threshold: {best_t} (update PD_CONDITIONAL_THRESHOLD in main.py if desired)")
