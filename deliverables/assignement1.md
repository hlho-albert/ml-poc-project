## Description du projet

The objective of this project is to identify potentially undervalued real estate properties using machine learning. The dataset is based on Idealista property listings and includes information about listed prices, property characteristics, locations, and descriptions. The goal is to estimate the theoretical market value of each property and compare it with its actual listed price.

## Définition du problème

This project will be treated as a regression problem. The model will predict either the total property price or the price per square meter. The predicted value will then be compared with the actual listed price to identify properties that may be undervalued.

## Description du dataset choisi

The dataset contains real estate listings from Idealista. Each row represents one property listing. The data includes property prices, descriptions, and several characteristics related to the property and its location.

## Description des features disponibles

The available features may include the listed price, surface area, number of rooms, number of bathrooms, location, district, property type, floor, elevator availability, condition of the property, and textual description. Some features may need to be cleaned or transformed before being used in the model.

## Premières analyses exploratoires — EDA

The first exploratory analysis will focus on understanding the dataset structure, checking missing values, identifying duplicates, analyzing price distributions, and detecting outliers. I will also study the relationship between price and key variables such as surface area, location, number of rooms, and price per square meter.

## Objectif business

The business objective is to help real estate investors quickly identify properties that appear to be listed below their estimated market value. This could be useful as a first screening tool to prioritize properties that deserve deeper financial, legal, and operational analysis.

## Contexte machine learning

The machine learning model will be trained on historical property listings where the listed price is known. Based on the available features, the model will learn patterns that explain property prices and then estimate a fair market value for each listing.

## Métrique ou fonction de coût envisagée

The main metric considered will be Mean Absolute Error because it is easy to interpret in euros. Other possible metrics include RMSE, R² score, and MAPE. These metrics will help evaluate how close the model predictions are to the actual listed prices.

## Hypothèses, risques et limites identifiées

The main hypothesis is that property characteristics such as location, size, number of rooms, and condition can explain a significant part of the listed price. However, the model has several limits: listed prices are not final transaction prices, some important factors may be missing from the dataset, and a property may look undervalued because of hidden issues such as legal problems, renovation costs, poor building condition, or unattractive micro-location.

## Données / notebooks

The dataset will be stored in the repository, most likely in a `data/` folder. The notebooks used for data cleaning, exploratory analysis, and modeling will be stored in a `notebooks/` folder. To use the project, the user will need to clone the repository, create a virtual environment, install the requirements, and run the notebooks.