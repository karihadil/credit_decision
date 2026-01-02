import joblib
import numpy as np
import pandas as pd
from collections import Counter
from urllib.parse import urlparse
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, fbeta_score, precision_score
from ipaddress import ip_address
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance, DMatrix
from xgboost.callback import EarlyStopping
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import optuna
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, recall_score, f1_score, average_precision_score
from sklearn.calibration import calibration_curve



df=pd.read_csv('accepted_imputed - copie.csv')
df.info()
print(df.columns)
print(df["loan_status"].value_counts())
def map_status(status):
    if status in ["Charged Off", "Default", "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off"]:
        return 1   # default
    elif status in ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]:
        return 0   # non-default
    else:
        return None  #empty
df["default_flag"] = df["loan_status"].apply(map_status)
print(df['default_flag'].isna().sum())
df = df.dropna(subset=["default_flag"])
df['default_flag'] = df['default_flag'].astype(int)
print(df["default_flag"].value_counts())

print(df.describe())
print(df.isnull().sum())
print(df['default_flag'].unique())
print(df.info())
print("empty values :",df.isnull().sum().sort_values(ascending=False).head(20))
print("unique values:",df.nunique().sort_values())
# 1. Drop obvious ID-like columns
def drop_id_columns(df):
    id_like = [col for col in df.columns if 'id' in col.lower()]
    return df.drop(columns=id_like), id_like

# 2. Drop constant and near-constant features
def drop_constant_features(df, threshold=0.999):
    const_cols = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
        if top_freq >= threshold:  # 99.9% of values the same
            const_cols.append(col)
    return df.drop(columns=const_cols), const_cols

# 3. Drop high cardinality categoricals 
def drop_high_cardinality(df, max_unique_ratio=0.2):
    high_card_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) > max_unique_ratio:
            high_card_cols.append(col)
    return df.drop(columns=high_card_cols), high_card_cols

# 4. Drop highly correlated numeric features
def drop_correlated(df, threshold=0.95):
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=correlated_cols), correlated_cols

# 5. Detect potential leakage 
def detect_leakage(df):
    leakage_cols = [col for col in df.columns if 
                    'settlement' in col.lower() or 
                    'recoveries' in col.lower() or
                    'default' in col.lower()]  # except target
    return leakage_cols


def feature_audit(df, target_col='default_flag'):
    dropped = {}

    df, dropped['id_like'] = drop_id_columns(df)
    df, dropped['constant'] = drop_constant_features(df)
    df, dropped['high_card'] = drop_high_cardinality(df)
    df, dropped['correlated'] = drop_correlated(df)
    dropped['potential_leakage'] = detect_leakage(df)
    if target_col in dropped['potential_leakage']:
        dropped['potential_leakage'].remove(target_col)

    return df, dropped

clean_df, dropped_info = feature_audit(df, target_col='default_flag')
clean_df = clean_df.drop(columns=dropped_info['potential_leakage'], errors="ignore")



print("Dropped Summary:")
for k, v in dropped_info.items():
    print(f"{k}: {len(v)} -> {v[:10]}")  # show first 10 if too long

print("Shape before:", df.shape)
print("Shape after:", clean_df.shape)
print(clean_df.info())
print(clean_df.isnull().sum())

df = clean_df.copy()
drop_features = [
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "policy_code",      
    "emp_title",        
    "zip_code",         
    "issue_d",          
    "hardship_reason",
    "hardship_status",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "grade", 
    "sub_grade",
    "hardship_amount",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "hardship_payoff_balance_amount",
    "orig_projected_additional_accrued_interest"
]
df = df.drop(columns=drop_features,errors="ignore")

df.drop("loan_status", axis=1, inplace=True)
print("duplicates= ", df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
X = df.drop("default_flag", axis=1)
y = df["default_flag"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
cat_features = X.select_dtypes(include=["object"]).columns
num_features = X.select_dtypes(exclude=["object"]).columns
cardinality = X[cat_features].nunique()
ohe_features = [col for col in cat_features if cardinality[col] <= 20]   
label_features = [col for col in cat_features if cardinality[col] > 20]  
print("One-hot encoding:", ohe_features)
print("Label encoding:", label_features)
ohe_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", ohe_encoder, ohe_features),
        ("ordinal", ordinal_encoder, label_features)
    ],
    remainder="passthrough"
)


X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

encoded_feature_names = preprocessor.get_feature_names_out()
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_full = XGBClassifier(
    learning_rate=0.0453,
    max_depth=3,
    subsample=0.723,
    colsample_bytree=0.959,
    min_child_weight=1,
    gamma=3.574,
    scale_pos_weight=scale_pos_weight,
    n_estimators=2000,
    objective="binary:logistic",
    random_state=42,
    eval_metric="auc",
    n_jobs=-1
)

xgb_full.fit(X_train_enc, y_train, eval_set=[(X_test_enc, y_test)], verbose=0)


feat_importances = pd.Series(xgb_full.feature_importances_, index=preprocessor.get_feature_names_out())
top_features = feat_importances.sort_values(ascending=False).head(25).index.tolist()
joblib.dump(top_features, "top_features.pkl")

X_train_top = pd.DataFrame(X_train_enc, columns=preprocessor.get_feature_names_out())[top_features]
X_test_top = pd.DataFrame(X_test_enc, columns=preprocessor.get_feature_names_out())[top_features]

w = 1.5       # best scale_pos_weight
t = 0.2       # best threshold

xgb_model = XGBClassifier(
    learning_rate=0.0453,
    max_depth=3,
    subsample=0.723,
    colsample_bytree=0.959,
    min_child_weight=1,
    gamma=3.574,
    scale_pos_weight=w,
    n_estimators=500,
    objective="binary:logistic",
    random_state=42,
    eval_metric="auc",
    n_jobs=-1
)

calib = CalibratedClassifierCV(xgb_model, method="isotonic", cv=5)
calib.fit(X_train_top, y_train)

y_proba = calib.predict_proba(X_test_top)[:, 1]
y_pred = (y_proba > t).astype(int)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
fitted_xgb = calib.calibrated_classifiers_[0].estimator
feat_importances = pd.Series(
    fitted_xgb.feature_importances_,
    index=X_train_top.columns
)

feat_importances = pd.Series(
    fitted_xgb.feature_importances_,
    index=X_train_top.columns
)

top_feats = feat_importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8,6))
sns.barplot(x=top_feats.values, y=top_feats.index, palette="viridis")
plt.title("Top 20 Feature Importances (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

print("Feature Importances:\n", top_feats)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, color="green", lw=2, label=f"PR curve (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label="Calibrated model")
plt.plot([0,1], [0,1], linestyle="--", color="gray", label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("Observed default rate")
plt.title("Calibration Curve (PD vs Observed Defaults)")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.hist(y_proba, bins=50, edgecolor='k')
plt.xlabel("Predicted Probability of Default (PD)")
plt.ylabel("Number of Borrowers")
plt.title("Distribution of Predicted PDs")
plt.show()
def ks_statistic(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks = max(tpr - fpr)
    return ks

ks = ks_statistic(y_test, y_proba)
print(f"KS Statistic: {ks:.3f}")
def decile_test(y_true, y_proba, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df["decile"] = pd.qcut(df["y_proba"], q=n_bins, labels=False, duplicates="drop")

    summary = df.groupby("decile").agg(
        predicted_pd=("y_proba", "mean"),
        observed_default=("y_true", "mean"),
        count=("y_true", "size")
    ).reset_index()

    return summary.sort_values("decile")

decile_results = decile_test(y_test, y_proba)
print(decile_results)
joblib.dump(calib, "calib_model.pkl")
joblib.dump(preprocessor, 'preprocessor.pkl')