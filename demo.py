import pandas as pd
import json

# 1. Load test dataset
df_test = pd.read_csv("test.csv")   # adjust path if needed

# 2. Drop target column (VERY IMPORTANT)
TARGET_COL = "loan_status"   # or default / y label
if TARGET_COL in df_test.columns:
    df_test = df_test.drop(columns=[TARGET_COL])

# 3. Keep ONLY features used by your API / model
api_features = [
    'avg_cur_bal', 'dti', 'acc_open_past_24mths', 'term',
    'inq_last_12m', 'max_bal_bc', 'desc',
    'mths_since_recent_inq', 'open_rv_24m', 'loan_amnt',
    'num_actv_rev_tl', 'mort_acc', 'total_bal_il', 'all_util',
    'verification_status', 'home_ownership', 'int_rate',
    'fico_range_low'
]

df_api = df_test[api_features]

# 4. Take a few rows (e.g. 5–10)
df_sample = df_api.sample(5, random_state=42)

# 5. Convert to API-ready JSON
requests_payload = df_sample.to_dict(orient="records")

# 6. Save to file
with open("credit_api_demo_requests.json", "w") as f:
    json.dump(requests_payload, f, indent=2)

print("✅ Demo requests saved to credit_api_demo_requests.json")
