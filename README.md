# Credit Decision Engine API

This is a **bank-grade credit decision engine** built with FastAPI.  
It combines a machine learning **PD (Probability of Default) model** with banking rules to produce **loan decisions**.

---

## Features

- **Eligibility rules**: Hard-stop checks (income, DTI, loan vs income)
- **PD model**: Predicts probability of default using a trained ML model
- **Risk grading**: Converts PD into bank-friendly risk grades (AAA → CCC)
- **LGD assignment**: Policy-based Loss Given Default per product
- **Expected Loss (EL)**: Computes `EL = PD × LGD × EAD`
- **Profitability check**: Compares EL to interest income
- **Decision matrix**: Approve, Conditional, Reprice, Reject
- **Reason codes**: Explains why a loan is approved or rejected

