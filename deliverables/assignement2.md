# Assignment 2 — Data Cleaning and Feature Engineering

## 1. Objective

The objective of this second assignment is to transform the raw Idealista dataset into a clean, structured and model-ready dataset.

This step focuses on:

- cleaning duplicated or invalid observations;
- converting data types;
- handling missing values;
- treating extreme outliers;
- creating new real estate features;
- selecting variables for the future machine learning model;
- exporting the transformed datasets to `data/processed/`.

The main notebook used for this step is:

notebooks/03_feature_engineering.ipynb

The reusable Python functions are implemented in:

src/data.py

The final model-ready dataset is saved as:

data/processed/idealista_model_ready_latest.csv

---

## 2. Source Dataset

The source file used for this cleaning and feature engineering step is:

data/raw/idealista_raw_listings_20260430_185730.csv

The initial dataset contained:

Initial rows: 500
Initial columns: 51
Duplicated listings removed: 0

No duplicated listings were detected using the `propertyCode` identifier.

---

## 3. Data Cleaning Steps

The cleaning pipeline includes the following steps:

### 3.1 Column Name Normalization

The raw Idealista API response contains nested columns with names such as:

parkingSpace.hasParkingSpace
priceInfo.price.priceDropInfo.formerPrice

These column names were normalized by replacing dots, spaces and hyphens with underscores.

Example:

parkingSpace.hasParkingSpace
became:
parkingSpace_hasParkingSpace

This makes the dataset easier to use in Python and machine learning pipelines.

---

### 3.2 Duplicate Removal

Duplicates were checked using the `propertyCode` column, which acts as the unique property identifier.

Result:
Duplicates removed: 0

No duplicated listings were found in the dataset.

---

### 3.3 Data Type Conversion

Several columns were converted to numerical format:

price
size
rooms
bathrooms
latitude
longitude
distance
numPhotos
price_per_m2

Boolean variables were also converted into structured 0/1 features, including:

hasLift
exterior
parkingSpace_hasParkingSpace
parkingSpace_isParkingSpaceIncludedInPrice
newDevelopmentFinished

The `floor` column was transformed into a numerical variable called:

floor_numeric

A missingness indicator was also created:

floor_missing

---

## 4. Missing Values

The dataset contains several columns with high missingness.

The columns with at least 80% missing values were:

newDevelopmentFinished: 495 missing values (99.00%)
parkingSpace_parkingSpacePrice: 483 missing values (96.60%)
priceInfo_price_priceDropInfo_formerPrice: 411 missing values (82.20%)
priceInfo_price_priceDropInfo_priceDropValue: 411 missing values (82.20%)
priceInfo_price_priceDropInfo_priceDropPercentage: 411 missing values (82.20%)

These columns were not used directly as core model features.

For boolean variables, the strategy was:

1. create a missingness indicator;
2. fill missing boolean values with `False`;
3. convert the result into a 0/1 variable.

Example:

has_lift
has_lift_missing

For categorical variables, missing values were filled with:

Unknown

For numerical variables in the model-ready dataset, missing values were filled using the median.

After final preprocessing, the model-ready dataset contained:

Total missing values: 0

---

## 5. Outlier Treatment

The raw dataset is concentrated in Madrid’s premium residential market, so outlier treatment was intentionally conservative.

The goal was not to remove luxury properties, but to remove observations that were too extreme or atypical for a first machine learning model.

The following broad sanity filters were applied:

price between €100,000 and €10,000,000
size between 20 m² and 1,000 m²
rooms between 0 and 10
bathrooms between 1 and 10
price_per_m2 between €1,000/m² and €30,000/m²

Results:

Rows before filtering: 500
Rows after outlier filtering: 489
Outliers removed: 11
Outliers removed percentage: 2.2%

Only 2.2% of the dataset was removed, which preserves most of the sample while reducing the influence of extreme values.

---

## 6. Feature Engineering

Several real estate-specific features were created.

### 6.1 Ratio Features

price_per_m2
rooms_per_100m2
bathrooms_per_room

These features help describe the property more precisely than raw size or room count alone.

### 6.2 Log Features

log_size
log_price
log_num_photos

Log transformations help reduce skewness in variables such as price, size and number of photos.

### 6.3 Floor Features

floor_numeric
floor_missing

The original `floor` variable was parsed into a numerical format. Missing floor information was preserved through a missingness indicator.

### 6.4 Amenity Features

has_lift
has_lift_missing
is_exterior
is_exterior_missing
has_parking
has_parking_missing
parking_included
parking_included_missing
new_development_finished
new_development_finished_missing
has_price_drop

These variables capture important property characteristics and also preserve whether some information was missing.

### 6.5 Text-Based Features

The listing description was used to create simple NLP-inspired features:

description_length
has_luxury_keywords
has_renovation_keywords
has_investment_keywords

These features capture whether the listing description contains words related to luxury, renovation or investment potential.

### 6.6 Location-Based Analytical Features

The following features were created for exploratory and investment scoring purposes:

district_median_price_per_m2
district_listing_count
discount_vs_district_median_pct
neighborhood_median_price_per_m2
neighborhood_listing_count

However, some of these variables are derived from the target variable `price`.

Therefore, they are useful for analysis and scoring, but they are excluded from the first model-ready feature set to avoid target leakage.

---

## 7. Selected Model Features

The final model-ready dataset contains:

Rows: 489
Columns: 35
Target variable: price
Selected features: 34

The selected features are:

size
rooms
bathrooms
latitude
longitude
distance
numPhotos
floor_numeric
rooms_per_100m2
bathrooms_per_room
log_size
log_num_photos
description_length
has_lift
has_lift_missing
is_exterior
is_exterior_missing
has_parking
has_parking_missing
parking_included
parking_included_missing
new_development_finished
new_development_finished_missing
has_price_drop
has_luxury_keywords
has_renovation_keywords
has_investment_keywords
floor_missing
propertyType
district
neighborhood
municipality
detailedType_typology
detailedType_subTypology

The target variable is:

price

---

## 8. Numerical and Categorical Features

The final dataset contains:

28 numerical features
6 categorical features

The categorical features are:

propertyType
district
neighborhood
municipality
detailedType_typology
detailedType_subTypology

These categorical variables will be encoded during the model training phase, most likely using one-hot encoding inside a preprocessing pipeline.

---

## 9. Target Leakage Prevention

A key modeling decision was to exclude target-derived features from the model-ready dataset.

The following variables were created for analysis but excluded from the first model training dataset:

price_per_m2
district_median_price_per_m2
discount_vs_district_median_pct
neighborhood_median_price_per_m2

These variables are derived from `price`, which is the target variable.

Using them directly as inputs to predict `price` would create target leakage and artificially inflate model performance.

They may still be used later for investment scoring after the model has generated a predicted fair price.

---

## 10. Justification of Choices

The cleaning and feature engineering choices were designed to create a robust dataset for real estate price prediction.

The model needs features that represent:

* physical property characteristics;
* location;
* amenities;
* listing quality;
* text signals;
* missingness information.

Property size, number of rooms, bathrooms and location are expected to be strong predictors of price. Amenities such as lift, parking and exterior orientation can also influence value.

Missingness indicators were kept because missing information can itself be informative. For example, if parking information is missing, this may indicate that parking is not available or not highlighted in the listing.

Outlier filtering was applied conservatively because the dataset contains premium and luxury properties. Removing too many expensive properties would damage the representativeness of the premium market segment.

---

## 11. Alternatives Considered

Several alternatives were considered during the cleaning and transformation process.

### 11.1 Keeping All Outliers

One option was to keep all 500 observations.

This was not retained for the first model-ready dataset because very extreme properties, such as ultra-luxury assets or very large homes, may distort model training.

Instead, a broad outlier filter was applied, removing only 11 rows.

---

### 11.2 Dropping All Rows with Missing Values

Another option was to remove every row containing missing values.

This was not retained because it would unnecessarily reduce the dataset size and remove useful information.

Instead, missing values were handled using imputation and missingness indicators.

---

### 11.3 Removing All Columns with Missing Values

This was also rejected because many partially missing columns still contain useful information.

For example, parking and lift variables may be useful even if they are not always available.

---

### 11.4 Using `price_per_m2` as a Model Input

This was rejected for the first model-ready dataset because `price_per_m2` is calculated using the target variable `price`.

Using it as a feature would create target leakage.

---

### 11.5 Using District Median Price per m² as a Model Input

This was also rejected for the first model-ready dataset because it is derived from the target variable.

It may be considered later only if computed in a leakage-safe way inside the training pipeline.

---

## 12. Expected Impact on the Models

The transformations are expected to improve model performance by:

* removing invalid or extreme observations;
* reducing noise in the dataset;
* making variables usable by machine learning algorithms;
* capturing real estate-specific relationships through ratio features;
* improving the representation of skewed variables through log transformations;
* preserving information from missing values through missingness indicators;
* representing location through district, neighborhood and geographic coordinates;
* adding descriptive power through text-based keyword features.

The final dataset should be suitable for training the three regression models planned for the next assignment:

Ridge Regression
Random Forest Regressor
Gradient Boosting Regressor

---

## 13. Output Files

The following files were generated:

data/processed/idealista_clean_full_20260505_095832.csv
data/processed/idealista_model_ready_20260505_095832.csv
data/processed/idealista_model_ready_latest.csv
data/processed/idealista_cleaning_metadata_20260505_095832.json

The main dataset for the next assignment is:

data/processed/idealista_model_ready_latest.csv


This file will be used in Assignment 3 for model training and evaluation.

---

## 14. Reproducibility

To reproduce this step:

1. Run the data collection notebook:

notebooks/01_data_collection.ipynb

2. Run the exploratory data analysis notebook:

notebooks/02_eda.ipynb

3. Run the feature engineering notebook:

notebooks/03_feature_engineering.ipynb

4. Verify that the following file exists:

data/processed/idealista_model_ready_latest.csv

5. The reusable cleaning and feature engineering functions are available in:

src/data.py

---