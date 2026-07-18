# Customer Churn Prediction

## Overview

This project builds a logistic regression model to predict customer churn for a B2B subscription business, using account-level features such as tenure, purchase history, number of sites, and whether the account has a dedicated account manager. The pipeline covers data exploration, feature engineering, model training, evaluation, and application of the trained model to a set of new, unseen customers.

## Dataset

The primary dataset (`customer_churn.csv`) contains 900 customer records across 10 columns, with no missing values. Each record includes the customer's name, age, total purchase amount, whether they have an account manager, years as a customer, number of sites they operate, onboarding date, location, company, and the target variable, `Churn` (1 if the customer churned, 0 if not). A second file, `new_customers_1.csv`, contains 6 additional customers with the same feature columns but no `Churn` label, representing real accounts the business wants predictions for.

## Exploratory Data Analysis

Before modeling, the data was inspected for structure and quality: data types, summary statistics, shape, and a missing-value check confirmed a clean dataset with no null values to handle. An `Age_Category` feature was engineered by binning customers into Child (0–14), Adult (15–54), and Senior (55+) groups. In practice, the dataset's age range runs from 22 to 65, so the Child bin never actually gets populated; this is a known limitation of the binning choice and worth revisiting if age-based segmentation is explored further.

Two visualizations were produced to explore churn drivers. A bar plot of churn rate by age category showed how churn likelihood shifts across the adult and senior groups, and a second bar plot compared churn rate between customers with and without an account manager, testing whether that service touchpoint has a measurable effect on retention.

## Feature Engineering and Preprocessing

Identifier and non-numeric columns that don't carry predictive signal for the model in their raw form (`Names`, `Onboard_date`, `Location`, `Company`, and the engineered `Age_Category`) were dropped, leaving numeric features: age, total purchase, account manager status, years as a customer, and number of sites. The remaining data was split 80/20 into training and test sets. Feature scaling was applied using `StandardScaler`, fitting the scaler on the training set and applying that same transformation to the test set, which puts all features on a comparable numeric range before training.

## Model

A logistic regression classifier (`scikit-learn`) was trained on the scaled training data. Logistic regression was chosen as a baseline model for this binary classification problem, offering a fast, interpretable starting point before considering more complex alternatives.

## Evaluation

The model was evaluated on the held-out test set using accuracy and a confusion matrix. Overall accuracy came out to 90%, but a closer read of the confusion matrix tells a more complete story:

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 142 | 6 |
| **Actual: Churn** | 12 | 20 |

Of the 32 customers who actually churned, the model correctly identified 20, for a recall of 62.5%. In other words, over a third of at-risk customers were missed and would not have been flagged for retention outreach under this model. Precision on the churn class came out to 77%, meaning most customers flagged as likely to churn genuinely did. This gap between a strong headline accuracy number and a weaker recall is a common pitfall in churn modeling, where the majority class (customers who stay) dominates the dataset and can make a model look more effective than it actually is at catching the outcome that matters most to the business. Improving recall, potentially by adjusting the classification threshold below the default 0.5 or by trying a different model, is a natural next step.

## Applying the Model to New Customers

With the model trained and evaluated, it was applied to `new_customers_1.csv`, a set of 6 real customer accounts with unknown churn status. The same preprocessing steps used during training, dropping identifier columns and applying the fitted `StandardScaler`, were applied to this new data before generating predictions, ensuring the new customers were represented in exactly the same feature space the model was trained on. The result is a `Churn_Prediction` column added to the customer records, giving the business a concrete, actionable list of which new accounts are at elevated risk of churning. It's worth noting that because these customers have no true label, no accuracy metric applies to this step; its value is purely in generating a usable prediction where none existed before.

## Tech Stack

Python, pandas, numpy, matplotlib, seaborn, scikit-learn (LogisticRegression, StandardScaler, train_test_split, and evaluation metrics).

## Future Improvements

Potential next steps include revisiting the age binning to better reflect the actual customer age distribution, exploring `Onboard_date` and `Location` as engineered features rather than dropping them outright, testing alternative models (such as random forest or XGBoost) for comparison against the logistic regression baseline, and tuning the classification threshold to improve recall on the churn class given the cost of missing at-risk customers in a real business setting.
