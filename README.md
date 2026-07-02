# Credit Default Classification — ANN & Ensemble Methods with SHAP Explainability

A comprehensive binary classification pipeline that predicts whether a loan applicant will default on their credit. The project covers the full ML workflow: domain-driven feature engineering, model-specific preprocessing (including WOE encoding), baseline comparison across 7 models, Optuna hyperparameter tuning, ANN, ensemble methods (Hard Voting, Soft Voting, Stacking), and SHAP-based model explainability.

> **Note:** This project was developed on a local machine with limited computational resources. Optuna trial counts, epoch sizes, and search ranges were intentionally kept small to reduce training time. On a more powerful machine or cloud GPU (e.g. Kaggle, Google Colab), increasing trials to 100+ and epochs to 100+ would produce better optimized results.

---

## Problem

Predict `TARGET` — a binary label indicating whether a loan applicant defaulted (1) or not (0). The dataset is imbalanced (~8% defaulters), which heavily influences model selection and evaluation choices.

---

## Pipeline

```
Raw Data
   ↓
Feature Engineering (building aggregations, financial ratios, behavioral flags)
   ↓
Missing Value Imputation
   ↓
Model-Specific Preprocessing
(WOE encoding, outlier capping, feature selection, VIF, scaling)
   ↓
Baseline Model Comparison (7 models) — evaluated with Gini Coefficient
   ↓
Hyperparameter Tuning with Optuna
   ↓
ANN Model (Optuna tuned)
   ↓
Ensemble Methods (Hard Voting, Soft Voting, Stacking)
   ↓
Final Model Comparison with Gap Analysis → CatBoost Selected
   ↓
Feature Importance + SHAP Analysis
   ↓
Final CatBoost Retrained on Selected Features
```

---

## Feature Engineering

### Building-Level Aggregations
AVG/MODE/MEDI variants combined into single features to reduce dimensionality:

| Feature | Description |
|---------|-------------|
| `FLOORSMAX / FLOORSMIN` | Average of 3 floor measurements |
| `FLOORS_RANGE` | Gap between max and min floors |
| `APARTMENTS_INEQUALITY` | AVG vs MODE difference — apartment size variance |
| `LIVING_RATIO` | Fraction of building that is residential |
| `LAND_USE_EFFICIENCY` | Total area / land area — density of land use |
| `BUILD_TO_USE_GAP` | Construction-to-occupancy lag |
| `ELEVATOR_PER_ENTRANCE` | Building quality/convenience proxy |

### Financial Ratios
| Feature | Description |
|---------|-------------|
| `CREDIT_TO_INCOME_RATIO` | How much credit relative to income |
| `ANNUITY_TO_INCOME_RATIO` | Monthly repayment burden |
| `EXTRA_CREDIT` | Amount borrowed above goods price — cash loan signal |

### Behavioral & Risk Flags
| Feature | Description |
|---------|-------------|
| `REGION_INSTABILITY` | Sum of all address mismatch flags |
| `RECENTLY_CHANGED_PHONE` | Phone changed within last year |
| `OLD_ID` | ID document older than 10 years — potential fraud signal |
| `CREDIT_HUNGRY` | More than 3 bureau requests per month |
| `FULLY_ASSET_OWNER` | Owns both car and property |
| `NO_ASSETS` | Owns neither car nor property |
| `SINGLE_PARENT` | Children present but single/separated/widowed |
| `CHILDREN_NO_FAMILY` | Children present but household size ≤ 1 |

---

## Model-Specific Preprocessing

| Model | Encoding | Feature Selection | Scaling | Special |
|-------|----------|-------------------|---------|---------|
| Logistic Regression | WOE encoding | Correlation + VIF | None | Normality check |
| KNN | LabelEncoder | Correlation + VIF | StandardScaler | IQR outlier capping with percentile fallback |
| Random Forest | LabelEncoder | None | None | — |
| XGBoost / LightGBM / CatBoost | LabelEncoder | None | None | — |
| CatBoost Custom | None | None | None | Native categorical features |
| ANN | LabelEncoder | None | StandardScaler | — |

---

## Models Compared

### Phase 1 — Baseline (Default Parameters)
7 models: Logistic Regression, Random Forest, KNN, XGBoost, LightGBM, CatBoost, CatBoost (native categoricals)

### Phase 2 — Optuna Tuned

| Model | Trials | CV Folds |
|-------|--------|----------|
| KNN | 30 | 5 |
| Random Forest | 10 | 5 |
| XGBoost | 40 | 5 |
| LightGBM | 40 | 3 |
| CatBoost | 40 | 5 |
| ANN | 15 | — |

### Phase 3 — Ensemble Methods
- **Hard Voting** — majority vote across LR, RF, XGBoost, LightGBM, CatBoost
- **Soft Voting** — averaged probabilities across same models
- **Stacking** — LR, RF, XGBoost, CatBoost, LightGBM as base learners; CatBoost Optuna as meta-model (`passthrough=True`, `cv=5`)

---

## Evaluation Metric

All models evaluated using the **Gini Coefficient** — standard metric in credit risk:

```
Gini = 2 × AUC − 1
```

## Model Selection

Final comparison sorted by:
1. **Gini Gap** (Train − Test) ascending — prioritize generalization
2. **Test Gini** ascending — then by raw performance

### Selected Model: CatBoost Optuna

---

## SHAP Explainability

After model selection, SHAP (SHapley Additive exPlanations) was applied to the CatBoost model to understand feature contributions:

- **TreeExplainer** used for efficient SHAP computation on tree-based models
- **Summary plot** shows directional impact of each feature on predictions
- Features with importance > 1% retained for final model retraining

This step goes beyond accuracy — it explains *why* the model makes each prediction, which is critical in credit risk where regulators require model transparency.

---

## Final Model

CatBoost retrained on SHAP-selected features only:
- Reduces noise from low-importance features
- Optuna retuned on the reduced feature set (n_trials=40)
- Cleaner, more interpretable production-ready model

---
