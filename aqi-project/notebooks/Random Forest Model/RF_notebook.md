EDA → Feature Engineering → Train-Test Split → Random Forest Model → Evaluation → Feature Importance

Steps:
1. Import Required Libraries
2. Select Features Based on EDA Insights
3. Encode Categorical Variables
4. Train-Test Split
5. Train Random Forest Model
6. Make Predictions
7. Evaluate Model Performance.

Formula Used - Recursive Binary Split

-> TREE BASED ENSEMBLE
-> SPLIT METRIC: GINI, ENTROPY, LOG_LOSS ( criterion = "entropy", )

_______________________________________________________________


Summary of each cell step by step:

**Cell 3 – Select important features**
-> What: Choose pollutant features that influence AQI category.
-> Purpose: Select input variables (pollutants + season) and target variable (AQI category).

**Cell 4 – Encode categorical variables**
Convert text values (Season, AQI Category) into numeric form.
Machine learning models work with numbers, not text.

**Cell 5 – Train-test split**
-> What: Split dataset into training and testing parts.

**Cell 6 – Create Random Forest model**
-> Initialize Random Forest with selected parameters.

**Cell 7 – Train the model**
-> Train model using training dataset.

**Cell 8 – Predict AQI category**
What: Model predicts AQI category using test dataset.
Purpose: Generate predictions for unseen data.

**Cell 9 – Evaluate accuracy**
**Cell 10 – Classification report**
**Cell 11 – Confusion matrix**

**Cell 12 – Feature importance** 
-> Most important predictors (usually PM2.5 & PM10).

**Cell 13 – Plot feature importance**
-> Understand which pollutant contributes most to AQI prediction.







