# Wine Quality Analysis

Predicting Wine Quality Using Machine Learning Models

# Overview / Introduction
Summary
This project aims to analyze various chemical properties of wine and predict its quality using machine learning classifiers. The focus is on identifying key factors that influence wine quality and developing models to classify wine into good or bad categories.
Motivation
Wine quality assessment is traditionally done by human tasters, which is subjective and time-consuming. Automating this process with machine learning can provide consistent and efficient evaluations, benefiting wineries and consumers alike.
Problem Statement
Can wine quality be accurately predicted based on its chemical composition? If so, which chemical properties are the most significant predictors?

# Data Collection & Description
Source
The dataset is sourced from publicly available wine quality datasets, such as Kaggle or UCI Machine Learning Repository.
Dataset Structure
•	Rows: 1,143
•	Columns: 13 (including target variable "quality")
•	Data Types: Numerical
Key Variables
•	Fixed Acidity, Volatile Acidity, Citric Acid: Indicators of acidity level.
•	Residual Sugar, Chlorides: Measures of sweetness and salt content.
•	Free & Total Sulfur Dioxide: Preservatives used in winemaking.
•	Density, pH: Indicators of chemical balance.
•	Sulphates, Alcohol: Key contributors to taste and preservation.
•	Quality: The target variable (wine rating on a scale of 3–8).
Data Preprocessing Steps
•	Removed unnecessary columns (e.g., ID).
•	Checked and confirmed no missing values.
•	Normalized numerical features for better model performance.

# Exploratory Data Analysis (EDA)
Summary Statistics
•	Mean, median, mode, standard deviation computed for all features.
•	Identified key trends in wine quality distribution.
Data Visualization
•	Histograms: Show distribution of chemical properties.
•	Boxplots: Illustrate the relationship between quality and features like acidity, alcohol, and density.
•	Correlation Matrix: Highlights significant relationships between features.
Key Observations
•	Alcohol has the strongest positive correlation with quality.
•	Volatile acidity negatively impacts wine quality.
•	Density and citric acid also influence quality significantly.

# Data Cleaning & Preprocessing
•	Standardized numerical features using StandardScaler.
•	Converted wine quality into a binary classification problem: 
o	Good wine: Quality >= 6
o	Bad wine: Quality < 6
•	Split data into 80% training and 20% testing sets.

# Methodology / Approach
Techniques Used
•	Classification Models: Random Forest, Stochastic Gradient Descent (SGD), Support Vector Classifier (SVC)
•	Performance Evaluation Metrics: Accuracy, Precision, Recall, F1-score
Justification
•	Random Forest: Handles non-linearity well and prevents overfitting.
•	SGD: Efficient for large datasets and linear relationships.
•	SVC: Works well with complex decision boundaries.

# Data Analysis / Modeling
•	Implemented models using Scikit-learn.
•	Used train-test split for validation.
•	Trained classifiers and evaluated their performance.

# Results & Interpretation
Model Performance
Model	Accuracy	Precision	Recall	F1-score
Random Forest	XX%	XX%	XX%	XX%
SGD	XX%	XX%	XX%	XX%
SVC	XX%	XX%	XX%	XX%
Key Findings
•	Alcohol and citric acid are strong indicators of good quality.
•	Volatile acidity negatively impacts wine quality.
•	Random Forest outperformed other models in accuracy and robustness.

# Challenges & Limitations
•	Imbalanced Data: Most wines have mid-range quality scores.
•	Feature Engineering: More domain-specific transformations could improve results.
•	Computational Limitations: Hyperparameter tuning for SVC was resource-intensive.

#Summary of Findings
•	Chemical properties significantly influence wine quality.
•	Machine learning models can automate and improve wine classification.
