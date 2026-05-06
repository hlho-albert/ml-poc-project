# Assignment 3 — Model Definition, Evaluation Protocol and Model Selection

## 1. Problem Definition

The machine learning problem is a supervised regression problem.

The goal is to predict the listing price of a real estate property based on its characteristics.

The target variable is:

price

The input features include numerical and categorical property variables such as:

size
rooms
bathrooms
latitude
longitude
distance
numPhotos
floor_numeric
amenity indicators
propertyType
district
neighborhood
municipality
detailedType_typology
detailedType_subTypology

The model output will later be used to estimate whether a listing appears undervalued, fairly priced or overvalued by comparing the predicted fair price with the actual listing price.

The business interpretation is:

predicted_price > actual_price → potentially undervalued
predicted_price ≈ actual_price → fairly priced
predicted_price < actual_price → potentially overvalued

However, the core machine learning task remains price prediction through regression.

---

## 2. Dataset Used

The model-ready dataset used for this assignment is:

data/processed/idealista_model_ready_latest.csv

The dataset contains:

Rows: 489
Columns: 35
Target variable: price
Input features: 34

The train/test split is:

Training set: 391 rows and 34 features
Test set: 98 rows and 34 features

This dataset was created during Assignment 2 after cleaning, missing value treatment, outlier filtering and feature engineering.

---

## 3. Evaluation Metric

The main evaluation metric is:

MAE — Mean Absolute Error

MAE is selected because it is directly interpretable in euros.

For example:

MAE = 462,609

means that the model prediction is wrong by approximately €462,609 on average.

This is especially useful for a real estate price prediction problem because the business user can directly understand the average prediction error in monetary terms.

Secondary metrics are:

RMSE — Root Mean Squared Error
R² — Coefficient of Determination

RMSE is useful because it penalizes large prediction errors more strongly than MAE.

R² is useful because it measures the proportion of variance in property prices explained by the model.

The main model selection criterion is:

Lowest test MAE

---

## 4. Evaluation Protocol

The evaluation protocol follows a standard supervised learning workflow.

First, the dataset is split into:

80% training data
20% test data

The training set is used for model training and cross-validation.

The test set is held out and used only for final model evaluation.

The cross-validation protocol is:

5-fold KFold cross-validation on the training set

The random state is fixed to:

42

This ensures that the results are reproducible.

The preprocessing pipeline is fitted only on the training folds during cross-validation and then on the full training set before final test evaluation.

---

## 5. Preprocessing Pipeline

The model pipeline separates numerical and categorical features.

Numerical features are processed using:

Median imputation
Standard scaling

Categorical features are processed using:

Most frequent imputation
One-hot encoding

The preprocessing is implemented inside a scikit-learn `Pipeline` and `ColumnTransformer`.

This is important because it prevents data leakage between the training and test sets.

---

## 6. Selected Models

Exactly three models were selected for this assignment:

1. Ridge Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

These three models were chosen to compare a simple linear baseline with two non-linear ensemble models.

---

## 7. Model 1 — Ridge Regression

### Description

Ridge Regression is a linear regression model with L2 regularization.

It serves as the baseline model for this project.

### Main Assumptions

Ridge Regression assumes that the relationship between the input features and the target variable is mostly linear after preprocessing.

It also assumes that regularization can help reduce overfitting when there are many encoded categorical variables.

### Expected Advantages

Ridge Regression has several advantages:

Fast to train
Simple to interpret
Good baseline model
Less prone to overfitting than standard linear regression
Works well with one-hot encoded categorical variables

### Expected Limitations

The main limitation is that real estate prices are often driven by complex non-linear effects.

For example, the effect of size may depend on the district, property type, floor, amenities and location.

Ridge Regression may therefore underperform compared to tree-based models.

### Adequacy with the Problem

Ridge Regression is appropriate as a first benchmark. It allows us to understand whether a simple linear model can capture enough of the pricing structure before using more complex models.

---

## 8. Model 2 — Random Forest Regressor

### Description

Random Forest Regressor is an ensemble model based on multiple decision trees.

Each tree is trained on a random subset of the data and features, and the final prediction is the average prediction of all trees.

### Main Assumptions

Random Forest assumes that property prices can be modeled through non-linear decision rules.

It also assumes that combining many trees reduces variance and improves robustness.

### Expected Advantages

Random Forest has several advantages:

Captures non-linear relationships
Handles feature interactions well
Works well on tabular data
More robust than a single decision tree
Can model complex real estate patterns

### Expected Limitations

The main limitations are:

Less interpretable than linear regression
Can overfit if not controlled
May struggle to extrapolate outside observed data
Can be slower than Ridge Regression

### Adequacy with the Problem

Random Forest is well suited for real estate price prediction because housing prices are influenced by many interacting variables such as size, location, amenities and property type.

---

## 9. Model 3 — Gradient Boosting Regressor

### Description

Gradient Boosting Regressor is an ensemble model that builds trees sequentially.

Each new tree tries to correct the errors made by the previous trees.

### Main Assumptions

Gradient Boosting assumes that the prediction errors of previous models can be progressively reduced by adding new weak learners.

It is particularly useful when the relationship between features and the target is complex and non-linear.

### Expected Advantages

Gradient Boosting has several advantages:

Strong predictive performance on tabular data
Captures non-linear effects
Captures feature interactions
Often outperforms simpler models
Good balance between bias and variance when tuned properly

### Expected Limitations

The main limitations are:

More sensitive to hyperparameters
Can overfit if too many trees are used
Less interpretable than Ridge Regression
Requires more careful tuning than Random Forest

### Adequacy with the Problem

Gradient Boosting is highly appropriate for real estate price prediction because property prices are influenced by non-linear and interacting effects. For example, the value of an additional square meter depends heavily on district, property type and luxury characteristics.

---

## 10. Justification for the Three Models

The three selected models provide a balanced comparison.

Ridge Regression = simple linear baseline
Random Forest Regressor = robust non-linear ensemble model
Gradient Boosting Regressor = advanced sequential ensemble model

This allows the project to compare:

Linear vs non-linear models
Simple vs complex models
Baseline vs ensemble methods
Bias-variance tradeoff

The selected models are appropriate for a real estate regression problem because they cover different levels of complexity while remaining explainable enough for an academic project.

---

## 11. Cross-Validation Results

The models were first evaluated using 5-fold cross-validation on the training set.

| Model                       |   CV MAE |  CV RMSE | CV R² |
| --------------------------- | -------: | -------: | ----: |
| Gradient Boosting Regressor | €384,270 | €629,174 |  0.79 |
| Random Forest Regressor     | €428,286 | €692,839 |  0.75 |
| Ridge Regression            | €471,112 | €672,660 |  0.76 |

The cross-validation results show that Gradient Boosting achieved the lowest average MAE.

This suggests that Gradient Boosting generalized better across the training folds than the other two models.

---

## 12. Test Set Results

The models were then evaluated on the held-out test set.

| Model                       | Test MAE | Test RMSE | Test R² |
| --------------------------- | -------: | --------: | ------: |
| Gradient Boosting Regressor | €462,609 |  €690,149 |  0.8202 |
| Random Forest Regressor     | €498,648 |  €785,570 |    0.77 |
| Ridge Regression            | €525,459 |  €841,350 |    0.73 |

The Gradient Boosting Regressor achieved:

Lowest test MAE
Lowest test RMSE
Highest test R²

Therefore, it was selected as the best model.

---

## 13. Best Model Selection

The selected best model is:

Gradient Boosting Regressor

It achieved:

Test MAE: €462,609
Test RMSE: €690,149
Test R²: 0.8202

The model explains approximately 82% of the variance in property prices on the test set.

The average absolute error is approximately €462,609.

Although this error may seem high, the dataset is concentrated in Madrid’s premium residential market, where many properties are priced above €1 million and the average property price is above €2 million. In that context, this error level is acceptable for a first version of the model.

---

## 14. Interpretation of Results

The results show that non-linear models outperform the linear baseline.

Gradient Boosting performed best because it can capture complex relationships between property characteristics and price.

This is consistent with the real estate market, where price is influenced by interactions between:

size
location
district
property type
bathrooms
number of rooms
amenities
listing quality

Ridge Regression performed worst on the test set, which suggests that a purely linear relationship is not sufficient to model property prices accurately.

Random Forest performed better than Ridge Regression, but slightly worse than Gradient Boosting.

---

## 15. Feature Importance Interpretation

The best model was the Gradient Boosting Regressor.

The feature importance analysis shows that the most important variables are:

size
log_size
distance
district_Barrio de Salamanca
longitude
bathrooms
rooms_per_100m2
log_num_photos
numPhotos
neighborhood_Recoletos
description_length

This is consistent with real estate market logic.

The strongest predictors are related to:

Property size
Location
Premium district effects
Physical characteristics
Listing quality

The importance of `size` and `log_size` confirms that property area is one of the strongest drivers of price.

The importance of `district_Barrio de Salamanca`, `longitude`, `distance`, and neighborhood variables confirms that location is also a major price driver.

The importance of `bathrooms`, `rooms_per_100m2`, and listing description variables suggests that property characteristics and listing quality also contribute to the model’s predictions.

---

## 16. Business Interpretation

The model can now estimate a fair market price for a property based on its characteristics.

This prediction can be compared with the actual listing price to identify possible opportunities.

For example:

If predicted price is significantly above actual price:
the property may be undervalued.

If predicted price is close to actual price:
the property may be fairly priced.

If predicted price is significantly below actual price:
the property may be overvalued.

This will be used in later stages to build an investment opportunity score.

The current model is not meant to make final investment decisions. Instead, it provides a first quantitative filter to identify properties that deserve further analysis.

---

## 17. Limitations

The model has several limitations.

First, the dataset contains only 489 model-ready observations. A larger dataset would likely improve model robustness.

Second, the dataset is concentrated in Madrid’s premium residential market. The model may not generalize well to lower-priced segments or other cities.

Third, the target variable is the Idealista listing price, not the final transaction price. Listing prices may differ from actual sale prices.

Fourth, some important real estate factors are not fully captured, such as:

property condition
renovation cost
building quality
exact floor plan
views
noise
legal constraints
negotiation potential

Fifth, the model can identify statistical underpricing, but it cannot confirm whether a property is truly a good investment without further due diligence.

---

## 18. Output Files

The notebook generated the following files:

models/best_model_latest.pkl
reports/model_metrics_latest.csv
data/processed/model_predictions_latest.csv
reports/model_training_metadata_latest.json

The best model file is:

models/best_model_latest.pkl

This file will be reused later in the Streamlit application.

The prediction file is:

data/processed/model_predictions_latest.csv

This file will be useful for visualizations and opportunity scoring.

---

## 19. Notebook and Reproducibility

The modeling notebook is located at:

notebooks/04_model_training.ipynb

To reproduce the experiments:

1. Run the data collection notebook:

notebooks/01_data_collection.ipynb

2. Run the exploratory data analysis notebook:

notebooks/02_eda.ipynb

3. Run the cleaning and feature engineering notebook:

notebooks/03_feature_engineering.ipynb

4. Run the model training notebook:

notebooks/04_model_training.ipynb

The model-ready dataset used is:

data/processed/idealista_model_ready_latest.csv

The random state is fixed to:
42

This ensures reproducible train/test splits and model results.

---
