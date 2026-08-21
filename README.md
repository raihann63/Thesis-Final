# FedStack: Federated Learning with Explainable AI for Population and Life Expectancy Prediction

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

##  Overview

**FedStack** is a research project that predicts **population** and **life expectancy** using a **federated stacked ensemble** framework. It combines:

- **Federated Learning** (privacy‑preserving, no raw data sharing)
- **Stacked Generalization** (Ridge meta‑learner on client predictions)
- **Explainable AI** (two‑level SHAP analysis)
- **Strict Leakage Prevention** (temporal OOF protocol, train‑only scaling/imputation)

The system is evaluated on **189 countries** (World Bank data, 2000–2024) under both **IID** and **realistic Non‑IID** (continent‑based) partitions.

---

##  Key Features

- **Privacy‑Preserving:** Clients share only predictions, not raw data.
- **Robust to Heterogeneity:** Ridge meta‑learner adapts to client data imbalances.
- **Leakage‑Free:** Time‑based split, train‑only statistics, backward‑looking features.
- **Interpretable:** Two‑level SHAP explains both client contributions and feature importance.
- **Reproducible:** All hyperparameters, seeds, and OOF folds are fixed.

---

##  Dataset

| Property | Value |
|----------|-------|
| **Source** | World Bank Open Data |
| **Countries** | 189 |
| **Time Range** | 2000–2024 |
| **Total Samples** | 4,725 |
| **Base Features** | 18 |
| **Engineered Features** | 32 (lag, rolling, polynomial, interaction, country stats) |
| **Total Features** | 50 |
| **Train Split** | 2000–2021 (4,158 rows) |
| **Test Split** | 2022–2024 (567 rows) |

---

## Methodology

### 1. Feature Engineering
- **Lags (1–3)** – capture temporal momentum.
- **Rolling windows (3‑yr, 5‑yr)** – smooth short‑term fluctuations.
- **Polynomial terms** – model non‑linear effects (GDP, fertility).
- **Interactions** – e.g., birth/death ratio.
- **Country‑level stats** – historical mean/std per country.

### 2. Federated Setup
- **5 clients** (IID: random shuffle / Non‑IID: continent‑based).
- **5 communication rounds** (empirically justified – no gain beyond 5).

### 3. FedStack Algorithm
- Each client trains a local **GradientBoostingRegressor**.
- Clients send **predictions** (not gradients) to the server.
- Server trains a **Ridge meta‑learner** on out‑of‑fold predictions.
- Final prediction = Ridge(GBM_pred₁, …, GBM_pred₅).

### 4. Leakage Prevention
- **Time‑based split** – no future data in training.
- **Strict Temporal OOF Protocol** – 19 expanding‑window folds; Ridge trained only on unseen‑by‑GBM predictions.
- **Train‑only statistics** – scaling, imputation, country stats fitted only on training set.
- **Backward‑looking features** – lags/rollings use only past values.

### 5. Explainability
- **Level 1 (Client Contribution):** `LinearExplainer` on Ridge meta‑learner.
- **Level 2 (Feature Importance):** `TreeExplainer` on GBM base models.

---

##  Results (Test Set 2022–2024)

```bash
git clone https://github.com/yourusername/fedstack.git
cd fedstack
