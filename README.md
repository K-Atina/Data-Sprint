# DataSprint 2026 — Kenya FinAccess Financial Status Prediction

**Strathmore Data Community × iLab Africa**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Environment & Dependencies](#3-environment--dependencies)
4. [Methodology](#4-methodology)
   - [Data Loading & Inspection](#41-data-loading--inspection)
   - [Exploratory Data Analysis (EDA)](#42-exploratory-data-analysis-eda)
   - [Preprocessing](#43-preprocessing)
   - [Modelling](#44-modelling)
   - [Evaluation](#45-evaluation)
   - [Feature Importance](#46-feature-importance)
5. [Key Findings](#5-key-findings)
6. [Model Performance](#6-model-performance)
7. [Recommendations](#7-recommendations)

---

## 1. Project Overview

This project was built for **DataSprint 2026**, a one-week data science challenge organised by the Strathmore Data Community (SDC) in collaboration with iLab Africa. The challenge uses the **2024 FinAccess Household Survey** with the goal being to build a machine learning model that predicts whether a Kenyan adult's financial situation has **Improved**, **Stayed the same**, or **Worsened** compared to the previous year, and to surface the key drivers behind financial deterioration.

---

## 2. Dataset

**Source:** [Kaggle — Kenya FinAccess Household Survey 2024](https://www.kaggle.com/datasets/davidpbriggs/kenya-finaccess-household-survey-2024)
**Official report & data manual:** [finaccess.knbs.or.ke](https://finaccess.knbs.or.ke/reports-and-datasets)
**File:** `finaccess2024_datasprint.xlsx` (also available as `.csv`) — **20,871 rows × 28 columns**

### Feature Categories

| Category | Columns |
|----------|---------|
| **Demographics** | `county`, `location_type`, `Sex`, `Age`, `household_size`, `education_level`, `marital_status`, `has_disability` |
| **Livelihood & Income** | `monthly_income`, `experienced_shock` |
| **Mobile & Digital** | `mobile_ownership_1`, `mobile_money_access`, `barriers_mobile_money` |
| **Financial Behaviour** | `Savings_formal`, `Savings_informal`, `Loan_formal`, `Loan_informal`, `defaulted`, `formal_service_use`, `barriers_bank`, `prodsum1` |
| **Financial Health & Literacy** | `fl_score`, `nfhi_11`, `nfhi_12`, `nfhi_13`, `accessto_13k_1month`, `not_difficult` |
| **Target** | `financial_status` |

### Target Class Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| Worsened | ~10,978 | 52.6% |
| Stayed the same | ~5,614 | 26.9% |
| Improved | ~4,279 | 20.5% |

> **Note:** The dataset is significantly imbalanced — over half of respondents reported a worsened financial situation. This was handled using `class_weight='balanced'` in both models and stratified train/test splitting.

### Missing Values

| Column | Missing % | Handling Strategy |
|--------|-----------|------------------|
| `barriers_bank` | 27.5% | Filled with `'No barrier'` — respondents without a missing value are those who *already have* a bank account, so no barrier applies |
| `monthly_income` | ~0% | Pre-imputed with median in the curated dataset; no action required |

---

## 3. Environment & Dependencies

The notebook was developed and tested in **Google Colab** (Python 3.10+).

### Required Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Installation (local environment)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running in Colab

Click the badge at the top of the notebook or go to:
`https://colab.research.google.com/github/K-Atina/Data-Sprint/blob/main/DataSprint.ipynb`

Upload `finaccess2024_datasprint.xlsx` to `/content/` before running all cells.

---

## 4. Methodology

### 4.1 Data Loading & Inspection

```python
df = pd.read_excel('/content/finaccess2024_datasprint.xlsx')
print(df.shape)          # (20871, 28)
print(df.dtypes)
df['financial_status'].value_counts()
```

Initial inspection confirmed:
- 20,871 rows and 28 columns
- Mix of categorical and numerical features
- Severe class imbalance in the target variable

---

### 4.2 Exploratory Data Analysis (EDA)

Eight visualisations were produced to understand the data before modelling:

#### Chart 1 — Target Variable Distribution
A bar chart showing the count and percentage for each of the three `financial_status` classes. Confirmed the severe class imbalance (Worsened: 52.6%, Stayed the same: 26.9%, Improved: 20.5%).

<img width="995" height="606" alt="image" src="https://github.com/user-attachments/assets/ca1eefc3-16cf-413f-89ed-79e33117cd2c" />

#### Chart 2 — Financial Shock vs. Financial Status
A grouped bar chart showing that among individuals who experienced a financial shock in the past year, the proportion with a *Worsened* financial situation was significantly higher than among those who did not. This established `experienced_shock` as a strong early-signal predictor.

<img width="1118" height="622" alt="image" src="https://github.com/user-attachments/assets/b4986a5e-3ffd-4508-a018-4acd8f4d30a3" />

#### Chart 3 — Monthly Income vs. Financial Status
A box plot (outliers removed) comparing the distribution of `monthly_income` (KES) across the three classes. Finding: minimal difference between "Stayed the same" and "Worsened", suggesting income alone is insufficient to distinguish these classes and other behavioural factors matter more.

<img width="1122" height="620" alt="image" src="https://github.com/user-attachments/assets/f05d134d-72b5-435a-90b2-0d22eee481e8" />

#### Chart 4 — Education Level vs. Financial Status
A grouped bar chart showing financial outcomes by education tier. Finding: as education level increases (from None through to University), the proportion of respondents in the "Worsened" category decreases. University-educated individuals show notably lower "Worsened" proportions.

<img width="1367" height="726" alt="image" src="https://github.com/user-attachments/assets/be9d49f7-24c6-4400-b23a-178be278a3b4" />

#### Chart 5 — Number of Financial Products vs. Financial Status
A violin plot of `prodsum1` (number of financial services used) by financial status. Finding: individuals whose situation improved tended to use more financial products, suggesting that access to diverse financial services correlates with better outcomes.
<img width="710" height="388" alt="image" src="https://github.com/user-attachments/assets/789f5bd3-f29f-4d49-988a-1698818303a7" />


#### Chart 6 — Location Type & Sex vs. Financial Status
A side-by-side grouped bar chart. Findings:
- Rural respondents show a slightly higher proportion in the "Worsened" category compared to urban.
- Women make up a slightly higher proportion of "Worsened" respondents compared to men, though the differences are small.
<img width="702" height="275" alt="image" src="https://github.com/user-attachments/assets/dae7bd51-ec06-4c4f-b242-734790097b6a" />


#### Chart 7 — Correlation Heatmap
A heatmap of correlations between the three numerical features: `monthly_income`, `household_size`, and `prodsum1`. Used to check for multicollinearity before modelling.

<img width="710" height="478" alt="image" src="https://github.com/user-attachments/assets/ff2ece45-983d-40b8-8c9d-a7a508c0fd92" />

---

### 4.3 Preprocessing

All preprocessing steps are applied to a copy of the dataframe (`df_model`):

1. **Missing value imputation** — `barriers_bank` filled with `'No barrier'` for respondents without missing values (i.e., those who already have bank accounts).

2. **Education label cleaning** — Verbose survey labels mapped to clean short-form labels (e.g., `'"University completed "'` → `'University'`).

3. **Minor-age row removal** — Rows where `Age == '16-17'` were dropped, as they fall outside the adult-policy scope.

4. **Marital status cleaning** — Ambiguous entries (`"Don't know"`, `"Refused to Answer"`) consolidated into `'Unknown'`.

5. **Feature engineering — Income band** — A new ordinal feature `income_band` was created from `monthly_income` using four real-world income tiers relevant to the Kenyan context:

   | Band | Range (KES/month) |
   |------|------------------|
   | Low | < 3,000 |
   | Lower-mid | 3,000 – 9,999 |
   | Mid | 10,000 – 29,999 |
   | High | 30,000+ |

6. **Categorical encoding** — All remaining categorical columns were label-encoded using `sklearn.preprocessing.LabelEncoder`. The ordinal `income_band` was encoded using `.cat.codes`.

7. **Target encoding** — `financial_status` mapped to integers: `Improved=0`, `Stayed the same=1`, `Worsened=2`.

8. **Train/test split** — 80/20 split with `stratify=y` to preserve class proportions in both sets and `random_state=42` for reproducibility.

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42
   )
   # Train size: ~16,696 | Test size: ~4,175
   ```

9. **Feature scaling** — `StandardScaler` applied to training data and used to transform (not fit) the test set. Required for Logistic Regression; not applied to the Random Forest input.

---

### 4.4 Modelling

Two models were trained and compared:

#### Model 1: Logistic Regression

```python
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_scaled, y_train)
```

- `class_weight='balanced'` automatically adjusts weights inversely proportional to class frequency, directly addressing the 52.6% / 26.9% / 20.5% imbalance.
- `max_iter=1000` ensures convergence on the full feature set.

#### Model 2: Random Forest (Selected Model)

```python
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

- 200 trees used for a stable importance estimate.
- Does not require feature scaling.
- `n_jobs=-1` uses all available CPU cores for faster training.
- Random Forest was selected as the final model because it achieved a higher Weighted F1-Score than Logistic Regression.

---

### 4.5 Evaluation

Primary metric: **Weighted F1-Score** — chosen because it accounts for class imbalance by weighting each class's F1 by its support (number of true instances).

```python
score = f1_score(y_test, y_pred_rf, average='weighted')
```

> A model that only predicts "Worsened" would achieve high accuracy (52.6%) but a low Weighted F1, since it would fail entirely on the other two classes.

Evaluation outputs produced:
- Full `classification_report` (precision, recall, F1, support per class)
- Confusion matrix heatmap
- Per-class F1 bar chart with weighted F1 reference line

**Confusion Matrix Interpretation:** The model is strongest at identifying "Worsened" (the majority class) but frequently misclassifies "Improved" and "Stayed the same" as "Worsened". The "Stayed the same" class was the hardest to predict (lowest per-class F1 ≈ 0.23), as it occupies the middle ground between the other two and gets pulled toward both.

---

### 4.6 Feature Importance

Random Forest's built-in `feature_importances_` attribute was used to rank all features. The top 15 most predictive features were visualised in a horizontal bar chart.

```python
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top15 = importances.sort_values(ascending=False).head(15)
```

---

## 5. Key Findings

### From EDA

| Finding | Insight |
|---------|---------|
| **Majority worsening** | 52.6% of Kenyan adults surveyed reported a deteriorating financial situation — the scale of financial stress is systemic, not individual |
| **Shocks drive deterioration** | Experiencing a financial shock (drought, illness, job loss) significantly raises the probability of a worsened outcome |
| **Income alone is insufficient** | Monthly income distributions barely differ between "Worsened" and "Stayed the same" — other factors (behaviour, access, shocks) matter more |
| **Education matters** | Higher education correlates with a lower probability of financial deterioration; respondents with no formal education are disproportionately represented in the "Worsened" group |
| **Financial product diversity helps** | Individuals who used more financial products (higher `prodsum1`) were more likely to report improved outcomes |
| **Gender & location gaps** | Rural residents and women are slightly more likely to report worsened outcomes, though the gaps are modest |

### From Feature Importance (Random Forest)

The model's top predictors of financial status are a combination of **financial health indicators**, **income**, and **financial behaviour** features. The most important single feature was `monthly_income`, but financial health variables (`nfhi_11` — food security, `nfhi_12` — non-food spending management, `nfhi_13` — absence of debt stress, `accessto_13k_1month` — emergency fund access) featured prominently, confirming that **financial resilience indicators are the strongest predictors** of whether a household's situation improved or deteriorated.

---

## 6. Model Performance

| Model | Weighted F1-Score |
|-------|------------------|
| Logistic Regression | 0.498 |
| **Random Forest** ✓ | **0.503** |

### Per-Class F1 Scores (Random Forest)

| Class | F1-Score |
|-------|---------|
| Improved | ~0.50–0.60 |
| Stayed the same | ~0.23 |
| Worsened | ~0.70+ |

---

## 7. Recommendations

Based on the model's feature importance rankings and EDA findings, the following actions are recommended for policymakers, banks, and NGOs:

### For Policymakers
- **Strengthen shock-response safety nets.** Financial shocks (drought, illness, job loss) are the clearest driver of financial deterioration. Targeted cash-transfer programmes, crop insurance, and health insurance for low-income households would directly reduce this risk.
- **Invest in education.** Higher education consistently correlates with better financial outcomes. Adult literacy and vocational training programmes in underserved counties can help break the education–poverty cycle.
- **Prioritise rural financial access.** Rural respondents report worse outcomes. Expanding mobile money infrastructure and agent banking into rural and remote areas reduces the access gap.

### For Banks & Financial Institutions
- **Develop products for the financially excluded.** 9.9% of Kenyan adults have no formal financial access. Low-documentation, mobile-first accounts (similar to M-Shwari) can onboard this segment.
- **Promote financial product bundling.** Respondents using more financial products reported better outcomes. Bundled savings + insurance + credit products can build household resilience.
- **Address the `barriers_bank` gap.** Affordability and eligibility remain the main barriers to bank access. Fee-free basic accounts and simplified KYC for low-income customers would reduce these barriers.

### For NGOs & Development Partners
- **Target interventions at the food-insecure.** `nfhi_11` (food security in the past 12 months) was among the top predictors. Households that were food-insecure are at high risk of financial deterioration — they represent a high-priority intervention group.
- **Focus on women and rural communities.** Gender and location gaps in outcomes, while modest, are consistent. Gender-sensitive financial literacy programmes and women-focused savings groups (chama strengthening) can close these gaps over time.
- **Promote emergency fund literacy.** `accessto_13k_1month` (ability to access KES 13,000 in an emergency) was a strong predictor of financial status. Programmes that help households build even small emergency reserves can materially improve resilience.
