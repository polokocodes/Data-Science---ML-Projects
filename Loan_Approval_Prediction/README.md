# Loan Prediction Pipeline

Predicting loan approval outcomes from applicant demographic, income, and credit history data, built end-to-end as a self-directed machine learning exercise.

## Overview

This project works through the full supervised learning workflow — exploratory data analysis, data cleaning, feature engineering, and classification modeling — on a real-world loan approval dataset. The goal was not just to produce a working model, but to build each stage independently and stress-test it, catching and fixing real bugs along the way (see **Debugging Notes** below) rather than following a reference implementation blindly.

## Dataset

- **Source:** `loan.csv` — 614 loan applications, 13 columns (`Loan_ID` + 11 features + `Loan_Status` target)
- **Target:** `Loan_Status` — binary, `Y` (approved) / `N` (denied); 422 approved / 192 denied (~69% approval rate)
- **Features:** applicant demographics (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`), financials (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`), `Credit_History`, and `Property_Area`
- **Missing values:** present in 6 of the 12 feature columns — `Gender` (13), `Married` (3), `Dependents` (15), `Self_Employed` (32), `LoanAmount` (22), `Loan_Amount_Term` (14), `Credit_History` (50)

## Exploratory Data Analysis

- Checked shape, dtypes, and missingness across all columns.
- Univariate distributions for the target and all categorical features (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`, `Credit_History`), built as grid subplots (`plt.subplots()`) rather than one figure per chart.
- Bivariate analysis — approval rate *by* category, not just category counts — to identify which features actually carry predictive signal ahead of modeling.
- **Key finding:** `Credit_History` stands out as the dominant predictor. Applicants with a credit history of `1` are approved at a dramatically higher rate than those with `0`, a pattern confirmed later by its correlation with the target and its consistent presence at the top of feature importance in every reference model.
- Numeric feature distributions (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`) plotted as histograms, revealing strong right-skew in income and loan amount — a small number of high-income applicants stretch the distribution and motivated the log-transformed features added during cleaning.

## Data Cleaning & Feature Engineering

- Dropped `Loan_ID` (identifier, no predictive value).
- Encoded `Loan_Status` to binary (`Y` → 1, `N` → 0).
- Normalized `Dependents` (`"3+"` → `"3"`), then imputed and cast to `int`.
- Imputed missing categorical features with the column mode, and missing numeric features with the column median (chosen over mean given the skew identified in EDA).
- Engineered four new features:
  - `TotalIncome` = `ApplicantIncome` + `CoapplicantIncome` — combined household income available for repayment.
  - `EMI` = `LoanAmount` / `Loan_Amount_Term` — proxy for monthly repayment burden.
  - `BalanceIncome` = `TotalIncome` − (`EMI` × 1000) — income remaining after the loan payment, a more direct risk signal than income or loan size alone.
  - `LoanAmount_log` and `TotalIncome_log` — log-transforms to compress the right-skew found in EDA.
- One-hot encoded remaining categoricals (`Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`) with `drop_first=True` to avoid redundant/multicollinear columns.
- Standardized all features with `StandardScaler` (fit on train, applied to test) ahead of modeling.

## Model Building

- **Logistic Regression** (baseline): trained on the scaled, cleaned feature set with an 80/20 train/test split (`random_state=42`).

**Confusion Matrix (test set):**

| | Predicted: Denied | Predicted: Approved |
|---|---|---|
| **Actual: Denied** | 18 | 25 |
| **Actual: Approved** | 2 | 78 |

- **Accuracy:** 78%
- **Denied-class recall:** ~42% (18/43) — the model misses over half of actual denials, defaulting toward the majority "approved" class.
- **Approved-class precision:** the 25 false positives (denied applicants predicted as approved) are the more costly error type for a lender, since they represent risk that the model failed to flag.

This recall gap on the minority class mirrors a pattern seen in an earlier customer churn project — models trained on imbalanced targets systematically underperform on the underrepresented class, and a confusion matrix (not accuracy alone) is what surfaces it.

## Debugging Notes

Two silent bugs were caught and fixed during this project, both worth documenting since neither raised an error and both would have gone unnoticed without deliberate checking:

1. **`Dependents` column wiped by `.map()`.** Using `.map({'3+': 3})` to normalize the `"3+"` category silently converted every *other* value in the column to `NaN` (`.map()` performs a total lookup — anything not in the dictionary becomes missing). The downstream mode-imputation step then filled all of those new `NaN`s with `3`, leaving the entire column as a single constant value with zero reported missing values. Fixed by switching to `.replace('3+', '3')`, which only swaps the matched value and leaves everything else untouched.

2. **Target leakage via a duplicate `target` column.** A `target` column created during EDA bivariate plotting (`Loan_Status` recoded to 0/1 for `groupby` averaging) persisted into the final cleaned dataset. Since it was never dropped from the feature set, the logistic regression model achieved 100% test accuracy — a red flag rather than a good result, since real approval data isn't perfectly separable. Diagnosed via a target correlation check (`.corr()['Loan_Status']`), which showed `target` at a correlation of exactly `1.0`. Fixed by dropping the column from the dataset entirely.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Environment:** Google Colab

## Future Improvements

- Train and compare remaining classifiers (decision tree, random forest, KNN, SVM, gradient boosting, naive Bayes) against the logistic regression baseline.
- Address the class imbalance driving weak denied-class recall — candidates include class weighting, resampling (SMOTE/undersampling), or adjusting the classification threshold.
- Hyperparameter tuning via `GridSearchCV` on the strongest-performing model.
- Cross-validation (5-fold) for more robust performance estimates than a single train/test split.
- Feature importance analysis once tree-based models are trained, to confirm `Credit_History` and the engineered income features drive predictions as EDA suggested.
