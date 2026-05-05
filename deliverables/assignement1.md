# Assignment 1 — Project Definition and Exploratory Data Analysis

## 1. Project Description

The goal of this project is to build a machine learning application that detects potentially undervalued real estate investment opportunities using property listing data collected from the Idealista API.

The project focuses on residential properties in Madrid, with an initial dataset strongly concentrated in Madrid’s premium residential market. The final application will estimate the fair market price of a property based on its characteristics and compare this estimated price with the actual listing price. This comparison will be used to identify whether a property appears undervalued, fairly priced, or overvalued.

The final output of the project will be a Streamlit application allowing users to either:
- input property characteristics manually and receive a price estimation and investment opportunity score;
- explore collected Idealista listings and identify the most attractive potential investment opportunities.

The project follows a complete machine learning workflow:
1. Data collection using the Idealista API
2. Raw data storage
3. Exploratory data analysis
4. Data cleaning and preprocessing
5. Feature engineering
6. Model training and evaluation
7. Model comparison
8. Streamlit application development

---

## 2. Business Objective

The business objective is to help real estate investors identify potential investment opportunities faster and more systematically.

Real estate investors usually need to manually compare many listings across different districts, price levels, property sizes and amenities. This process is time-consuming and subjective.

This project aims to answer the following business question:

> Is this property priced below, above, or in line with similar properties in the same market?

The final tool should help users:
- estimate the fair market value of a property;
- detect potential underpriced listings;
- compare properties across districts;
- prioritize listings for deeper due diligence;
- support investment decisions with data.

At this stage, the project does not claim to make final investment decisions. Instead, it provides a first analytical filter to identify properties that may deserve further investigation.

---

## 3. Machine Learning Problem Definition

The main machine learning problem is a supervised regression problem.

The target variable is:

"price"

The model will learn to predict the listing price of a property based on its characteristics, such as:

size, rooms, bathrooms, district, neighborhood, property type, floor, lift, parking, latitude, longitude, and other listing features.

After predicting the fair price of a property, the model output will be compared with the actual listing price.

The business logic will be:

predicted_price > actual_price → potential undervalued opportunity
predicted_price ≈ actual_price → fairly priced property
predicted_price < actual_price → potentially overvalued property

A secondary business classification will be created from the regression output:

undervalued / fairly priced / overvalued

However, the core machine learning task remains regression.

---

## 4. Dataset Description

The dataset was collected from the Idealista API using official API credentials.

The initial raw dataset contains:

500 property listings
54 columns
0 duplicated listings
24 districts

The collected listings correspond to residential properties for sale in and around Madrid.

The dataset is stored in:

data/raw/

The raw CSV file follows this naming convention:

idealista_raw_listings_YYYYMMDD_HHMMSS.csv

The data collection notebook is located in:

notebooks/01_data_collection.ipynb

The exploratory data analysis notebook is located in:

notebooks/02_eda.ipynb

An EDA-enriched dataset is exported to:

data/processed/

with the following naming convention:

idealista_eda_enriched_YYYYMMDD_HHMMSS.csv

---

## 5. Data Collection Method

The data was obtained through the Idealista API.

The API workflow is the following:

1. Retrieve an OAuth access token using the API key and client secret.
2. Use the access token to call the Idealista Search API.
3. Collect property listings page by page.
4. Save raw API responses as JSON files.
5. Normalize the API response into a tabular pandas DataFrame.
6. Save the flattened listings as CSV files.

The API credentials are stored locally in a `.env` file and are not pushed to GitHub.

Example local `.env` structure:

IDEALISTA_API_KEY=your_api_key
IDEALISTA_CLIENT_SECRET=your_client_secret

The `.env` file is excluded from version control through `.gitignore`.

Due to compatibility issues between Python `requests` and the Idealista API response format in the local environment, the notebook uses a `curl`-based fallback to reproduce the HTTP request format that successfully authenticates and retrieves API data.

---

## 6. Available Features

The initial dataset includes property-level information, location information, listing metadata and some amenities.

Important available features include:

propertyCode
price
propertyType
operation
size
rooms
bathrooms
address
province
municipality
district
neighborhood
latitude
longitude
url
distance
numPhotos
floor
exterior
hasLift
parkingSpace.hasParkingSpace
parkingSpace.isParkingSpaceIncludedInPrice
newDevelopmentFinished

Some additional nested API fields are also available, especially related to price information, parking and development status.

The most important features for the initial machine learning model are expected to be:

size
rooms
bathrooms
district
neighborhood
latitude
longitude
propertyType
floor
hasLift
parking
numPhotos

A first engineered feature was also created during EDA:

price_per_m2 = price / size

This feature is not the final target, but it is highly useful for market analysis and identifying possible underpriced listings.

---

## 7. Exploratory Data Analysis

The initial EDA was performed in:

notebooks/02_eda.ipynb

### 7.1 Dataset Size

The raw dataset contains:

Rows: 500
Columns: 54
Duplicated propertyCode: 0

No duplicated listings were detected using the `propertyCode` identifier.

---

### 7.2 Price and Size Summary

The initial numerical summary is:

Median price: €1,787,500
Average price: €2,268,731
Minimum price: €300,000
Maximum price: €15,000,000

Median size: 186 m²
Average size: 269.46 m²
Minimum size: 37 m²
Maximum size: 1,700 m²

Median price per m²: €8,656/m²
Average price per m²: €9,488/m²

These results show that the dataset is strongly oriented toward premium and luxury residential properties.

---

### 7.3 District Distribution

The most represented districts are:

Barrio de Salamanca: 132 listings (26.40%)
Centro: 63 listings (12.60%)
Chamberí: 57 listings (11.40%)
Chamartín: 42 listings (8.40%)
Retiro: 32 listings (6.40%)
Moncloa: 29 listings (5.80%)
Hortaleza: 28 listings (5.60%)

The most represented district is Barrio de Salamanca, which accounts for more than one quarter of the dataset.

This confirms that the initial dataset is not representative of the entire Madrid housing market. Instead, it is concentrated in prime and high-value residential areas.

---

### 7.4 Price per Square Meter by District

Median price per square meter varies significantly across districts:

Barrio de Salamanca: €12,878/m²
Chamberí: €10,052/m²
Retiro: €9,798/m²
Chamartín: €9,453/m²
Centro: €8,074/m²
Arganzuela: €7,671/m²
Tetuán: €7,407/m²
Moncloa: €6,609/m²
Hortaleza: €6,255/m²

This confirms that location is one of the strongest drivers of property value.

District-level and neighborhood-level features will therefore be very important for the machine learning model.

---

### 7.5 Missing Values

Some columns have a high percentage of missing values:

newDevelopmentFinished: 99.00% missing
parkingSpace.parkingSpacePrice: 96.60% missing
price drop information fields: 82.20% missing
detailedType.subTypology: 68.00% missing
parkingSpace.hasParkingSpace: 51.80% missing
parkingSpace.isParkingSpaceIncludedInPrice: 51.80% missing
floor: 20.00% missing
exterior: 19.00% missing
hasLift: 18.40% missing
neighborhood: 8.40% missing

These missing values will need to be handled during the cleaning and preprocessing phase.

Possible strategies include:

* dropping columns with extremely high missingness;
* converting some missing values into meaningful binary indicators;
* imputing missing numerical or categorical values;
* keeping missingness indicators where absence of data may carry information.

---

### 7.6 Outliers

The dataset contains several high-end outliers:

Maximum price: €15,000,000
Maximum size: 1,700 m²
Maximum rooms: 9
Maximum bathrooms: 11

These values may be legitimate luxury properties, but they can distort model training.

During the cleaning and preprocessing phase, the project will test strategies such as:

* keeping outliers but using robust models;
* applying log transformation to the target variable;
* filtering extreme values for certain experiments;
* comparing model performance with and without extreme outliers.

---

## 8. First Investment Opportunity Proxy

A first exploratory investment indicator was created using price per square meter.

The logic is:

district_avg_price_per_m2 = average price per m² in the district

discount_vs_district_avg_pct =
(property price per m² - district average price per m²)
/
district average price per m²

Properties with a strongly negative value may be potential underpriced opportunities.

Example of top potential opportunities from the first EDA:

Moncloa property:
Price: €5,000,000
Size: 1,700 m²
Price per m²: €2,941/m²
Discount vs district average: -57.27%

Barrio de Salamanca property:
Price: €530,000
Size: 90 m²
Price per m²: €5,889/m²
Discount vs district average: -54.86%

Retiro property:
Price: €320,000
Size: 67 m²
Price per m²: €4,776/m²
Discount vs district average: -54.22%

This first proxy is useful for exploration, but it is still simplistic.

It does not yet account for:

* exact micro-location;
* property condition;
* floor level;
* renovation needs;
* amenities;
* luxury characteristics;
* property typology;
* data quality issues.

The final model will improve this by predicting fair market value using multiple features.

---

## 9. Machine Learning Context

The project will use supervised machine learning to estimate the fair market price of a property.

The regression model will be trained on historical listing data collected from Idealista. Each row represents a property listing, and the target variable is the listing price.

The expected modeling workflow is:

1. Clean the raw dataset.
2. Remove or handle duplicates.
3. Handle missing values.
4. Create relevant real estate features.
5. Encode categorical variables.
6. Split the data into training and testing sets.
7. Train exactly three regression models.
8. Evaluate the models using regression metrics.
9. Select the best-performing model.
10. Use the model predictions to create an investment opportunity score.

The planned models are:

1. Ridge Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor or XGBoost Regressor

The final choice between Gradient Boosting and XGBoost will depend on package availability and model performance.

---

## 10. Evaluation Metric

The main evaluation metric will be:
MAE — Mean Absolute Error
MAE is selected because it is easy to interpret in business terms.

Additional metrics will also be calculated:

RMSE — Root Mean Squared Error
R² — Coefficient of Determination

RMSE is useful because it penalizes large errors more strongly. R² is useful to understand how much variance in property prices is explained by the model.

However, MAE will remain the main metric because it is the most directly interpretable for real estate price estimation.

---

## 11. Initial Hypotheses

The initial hypotheses are:

1. Property size is one of the strongest predictors of price.
2. District and neighborhood have a major impact on price per square meter.
3. Premium districts such as Barrio de Salamanca, Chamberí and Retiro have significantly higher price per square meter.
4. Amenities such as lift, parking and exterior orientation may increase property value.
5. Properties priced significantly below their district average price per square meter may represent potential investment opportunities.
6. High-end outliers may reduce model stability if not handled carefully.
7. A non-linear model such as Random Forest or Gradient Boosting will likely outperform a simple linear regression model.

---

## 12. Risks and Limitations

The current dataset has several limitations.

First, the sample size is limited to 500 listings. This may be enough for an initial academic project, but a larger dataset would improve model robustness.

Second, the dataset is strongly concentrated in Madrid’s premium residential market. The model may therefore not generalize well to the entire Madrid housing market or to lower-priced segments.

Third, some important variables have high missingness, especially parking-related variables, new development information and price drop information.

Fourth, the listing price is not the same as the final transaction price. Idealista prices are asking prices, not necessarily closing prices.

Fifth, the first opportunity indicator is based on district-level average price per square meter. This is useful for early exploration, but it does not fully capture micro-location, property condition, floor, renovation status or negotiation potential.

Finally, some properties may be atypical luxury assets, which makes direct comparison difficult. The next steps will include outlier treatment, feature engineering and model evaluation to reduce this risk.

---

## 13. Data and Notebook Reproducibility

The project repository is organized as follows:

ml-poc-project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── deliverables/
│   ├── assignment1.md
│   ├── assignment2.md
│   ├── assignment3.md
│   ├── assignment4.md
│   └── assignment5.md
│
├── models/
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── 02_eda.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore

To reproduce the current work:

1. Create a `.env` file with the Idealista API credentials.
2. Run `notebooks/01_data_collection.ipynb` to collect the raw data.
3. Confirm that the raw CSV file is saved in `data/raw/`.
4. Run `notebooks/02_eda.ipynb` to perform exploratory data analysis.
5. Confirm that the EDA-enriched dataset is saved in `data/processed/`.

---

## 14. Next Steps

The next phase of the project will focus on data cleaning and feature engineering.

The main tasks will be:

1. Remove duplicated listings if any appear in future data pulls.
2. Handle missing values.
3. Decide which high-missingness columns to keep or remove.
4. Clean outliers.
5. Create final model features.
6. Encode categorical variables.
7. Prepare the model-ready dataset.
8. Save the transformed dataset in data/processed/.

The next notebook will be:

notebooks/03_feature_engineering.ipynb

The next source file to develop will be:

src/data.py

