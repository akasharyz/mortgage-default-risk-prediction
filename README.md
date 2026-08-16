# Mortgage Default Risk Prediction

Predicting loan default risk from application-time data using the Home Credit Default Risk dataset.

## Problem Framing

Lenders need to assess the risk of default *before* a loan is approved, using only the information available at application time. This project builds and compares three models — logistic regression, XGBoost, and a neural network — to predict whether an applicant will default, and evaluates them using metrics appropriate for a highly imbalanced target rather than plain accuracy.

**Why this is a real problem, not just a classification exercise:** only 8.1% of applicants in this dataset default. A model that predicts "no default" for every applicant would be 91.9% accurate and completely useless. The core technical challenge here isn't fitting a model — it's building one that's actually evaluated and optimized for the minority class that matters.

## Dataset

- **Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- **Table used:** `application_train.csv`
- **Size:** 307,511 applicants × 122 raw columns
- **Target:** `TARGET` — 1 if the applicant defaulted, 0 otherwise
- **Class balance:** 91.9% repaid / 8.1% defaulted

The dataset also includes auxiliary tables (`bureau.csv`, `previous_application.csv`, `installments_payments.csv`, etc.) which were **not used** in this iteration — see Limitations.

## Approach

### 1. Exploratory Data Analysis

- **Missingness:** 67 of 122 columns had missing values. Property/building descriptor columns (`COMMONAREA_*`, `LIVINGAPARTMENTS_*`, `FLOORSMIN_*`, `YEARS_BUILD_*`, etc.) were missing together at ~60–70% — evidence they represent one underlying theme (verifiable property records) rather than 30+ independent signals.
- **`OWN_CAR_AGE`:** missing in 66% of rows, but confirmed to be missing *exactly* when `FLAG_OWN_CAR = 'N'` — i.e., missing means "does not own a car," not "unknown."
- **Missingness as a signal:** applicants who defaulted had a higher median count of missing fields (48 vs. 34) than those who repaid — a candidate feature in its own right.
- **`DAYS_EMPLOYED` data bug:** the column contained a sentinel value of `365243` (≈1,000 years) in 55,374 rows (18% of the dataset) — 99.96% of them `NAME_INCOME_TYPE = 'Pensioner'`. This group has a materially *lower* default rate (5.4% vs. 8.7% for everyone else), confirming the anomaly is informative, not junk. It was replaced with `NaN` and preserved as a separate `DAYS_EMPLOYED_ANOMALY` flag.
- **Strongest numeric predictors:** `EXT_SOURCE_1/2/3` (external credit-bureau-style scores) showed the clearest separation between classes of any raw feature (e.g., `EXT_SOURCE_3` median: 0.546 for repaid vs. 0.379 for defaulted).
- **Categorical signal:** default rate fell monotonically with education level (10.9% for "Lower secondary" down to 1.8% for "Academic degree"), and varied meaningfully by occupation (17.2% for Low-skill Laborers vs. 4.8% for Accountants) — with a caution that small categories (n < ~500) produce unreliable rate estimates.

### 2. Feature Engineering

| Feature(s) | Reasoning |
|---|---|
| `DAYS_EMPLOYED_ANOMALY` + cleaned `DAYS_EMPLOYED` | Isolates the sentinel-value bug found in EDA while preserving its predictive signal |
| `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_ANNUITY_RATIO` | Loan burden relative to income, not absolute amounts |
| `AGE_YEARS`, `YEARS_EMPLOYED`, `EMPLOYED_AGE_RATIO` | Interpretable units; employment tenure relative to age |
| `OWN_CAR_AGE` filled with 0 | Missing = no car owned, not an unknown value — mean/median imputation would have been wrong |
| `PROPERTY_INFO_COUNT` + retained `_AVG` columns (dropped `_MODE`/`_MEDI`) | Collapsed 47 highly redundant, jointly-missing property columns into one count feature plus a non-redundant representative subset, after confirming the count carries real signal (20.0 vs. 17.0 mean fields provided, repaid vs. defaulted) |
| `EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`, `EXT_SOURCE_COUNT` | Combined signal from the three strongest predictors; `STD` captures disagreement between sources; `COUNT` documents how many of the three were actually available per applicant (only 35.6% of applicants have all three) |
| `CREDIT_GOODS_RATIO` (replacing `AMT_GOODS_PRICE`) | `AMT_GOODS_PRICE` and `AMT_CREDIT` were found to be severely collinear (r = 0.987), producing unstable, opposite-signed coefficients in logistic regression; replaced with a ratio |
| Rare-category bucketing (`ORGANIZATION_TYPE`, threshold < 500 samples → `"Other"`) | Small categories (some with fewer than 20 samples) were found to produce inflated, overfit coefficients in the linear model |

### 3. Train / Validation / Test Split

Stratified 70/15/15 split on `TARGET`, preserving the 8.07% default rate identically across all three sets (train: 215,257 rows, val/test: 46,127 rows each).

### 4. Models

| Model | Key configuration |
|---|---|
| Logistic Regression | `class_weight='balanced'`, median imputation, standard scaling, one-hot encoding |
| XGBoost | `scale_pos_weight ≈ 11.4`, `max_depth=5`, `learning_rate=0.05`, early stopping on validation PR-AUC |
| PyTorch MLP | 2 hidden layers (128 → 64), BatchNorm + Dropout(0.3), `BCEWithLogitsLoss` with `pos_weight`, best-epoch checkpointing |

### 5. Evaluation Metric

**AUC-ROC and Precision-Recall AUC (PR-AUC)** — not accuracy, given the 8.1% base rate. PR-AUC in particular is reported relative to the random-classifier baseline (~0.081, matching the default rate), since raw PR-AUC values are not interpretable on their own with this level of imbalance.

## Results

| Model | AUC-ROC | PR-AUC | PR-AUC vs. random baseline |
|---|---|---|---|
| Logistic Regression | 0.7503 | 0.2369 | ~2.9x |
| **XGBoost** | **0.7629** | **0.2534** | **~3.1x** |
| Neural Net (MLP, best epoch) | 0.7522 | 0.2455 | ~3.0x |

**XGBoost was the best-performing model on both metrics.** The MLP required explicit early-stopping/checkpointing to avoid overfitting (validation performance peaked around epoch 5–8 and degraded afterward while training loss kept improving) and still didn't surpass XGBoost even at its best epoch. This is consistent with the broader pattern that gradient-boosted trees tend to outperform neural networks on structured/tabular data with a mix of numeric and high-cardinality categorical features.

### Feature Importance (XGBoost)

The top predictors were, in order: `EXT_SOURCE_MEAN`, `EXT_SOURCE_COUNT`, education level (Higher education), `CODE_GENDER`, `FLAG_OWN_CAR`, `YEARS_EMPLOYED`, and the engineered `CREDIT_GOODS_RATIO` / `CREDIT_ANNUITY_RATIO` ratios. This is consistent across three independent checks — EDA correlations, logistic regression coefficients, and XGBoost feature importance all agree that external credit scores dominate.

**Fairness flag:** `CODE_GENDER` appeared in the top 5 most important features. In many jurisdictions, using gender directly in credit decisions is legally restricted under fair-lending regulations, and even where it isn't, using it uncritically risks encoding societal bias rather than genuine risk signal the model can't distinguish between the two. This project retains gender for analysis and transparency, but flags it explicitly as something a production model would need to remove and/or audit for disparate impact before deployment.

## Limitations

- **Single-table model.** Only `application_train.csv` was used. The auxiliary tables (credit bureau history, previous Home Credit loans, installment payment behavior) almost certainly contain additional signal — top solutions on this dataset that incorporate those tables report meaningfully higher AUC-ROC (~0.78–0.80) than achieved here.
- **Partial external scores.** Only 35.6% of applicants have all three `EXT_SOURCE` values; the rest are computed from 1–2 sources (or none, for 172 rows), meaning `EXT_SOURCE_MEAN` isn't computed identically across all applicants.
- **Unreliable rates for small categories.** Several categorical levels (e.g., certain `NAME_INCOME_TYPE` and `ORGANIZATION_TYPE` values) have too few samples for their observed default rates to be trusted; these were bucketed for modeling but the underlying uncertainty remains.
- **Fairness not audited.** `CODE_GENDER` was retained as a feature and ranks among the most important; no disparate-impact or fairness audit was performed on model outputs across gender or other protected attributes.
- **No SHAP/local explainability included in this iteration.** Feature importance is reported at the global level (XGBoost gain) only.
- **No hyperparameter tuning.** All three models used reasonable defaults rather than a tuned search (e.g., grid/random/Bayesian search over XGBoost depth, learning rate, regularization).

## Google Collab Notebook Link
[Google Collab Notebook](https://colab.research.google.com/drive/1pk-Y-tHQk6n-vOI0cyASHni-lhF08RwL#scrollTo=Kj5sZVi90qOA)

## Repository Structure

```
├── Home_Credit_ML.ipynb        # Full analysis: EDA → feature engineering → modeling → evaluation
├── README.md              # This file
└── data/                  # application_train.csv (not included — see Dataset section for source)
```
