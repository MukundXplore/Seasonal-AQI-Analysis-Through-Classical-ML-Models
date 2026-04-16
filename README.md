# 🌫️ Air Quality Index (AQI) Classification using Machine Learning

## 📌 Project Overview

This project focuses on predicting **Air Quality Index (AQI) categories** using supervised machine learning models based on pollutant concentrations and contextual features such as **city** and **season**.

The objective was to build **accurate, robust, and interpretable models**, while performing comprehensive evaluation using multiple metrics and validation techniques.

---

## 📊 Dataset

The dataset consists of air pollution measurements collected across multiple Indian cities (2015–2023), including:

### 🔹 Features

* **Pollutants:** PM2.5, PM10, NO₂, SO₂, CO, O₃
* **Contextual:** City, Season (engineered from date)

### 🔹 Target

* AQI Category (Good, Satisfactory, Moderate, Poor, etc.)

### 🔹 Preprocessing Steps

* Handling missing values and validating imputation integrity
* Feature selection and consistency checks
* One-hot encoding of categorical variables
* Feature scaling (StandardScaler for SVM)
* Validation of physical constraints (e.g., PM2.5 ≤ PM10)
* Dataset standardization and cleaning

---

## ⚙️ Models Implemented

### 1️⃣ Support Vector Machine (SVM)

* Kernel: RBF
* Hyperparameters: C = 10, gamma = scale
* Applied on scaled features
* Achieved strong generalization and stable performance

---

### 2️⃣ Random Forest

* Ensemble-based classifier
* Low variance and strong stability
* Used for feature importance analysis

---

### 3️⃣ Multiple Linear Regression (MLR)

* Indirect approach:

  * Predict AQI (numerical)
  * Map predictions → AQI categories
* High numerical performance but less suitable for classification

---

## 📈 Evaluation Metrics

To ensure fair and reliable evaluation:

* Accuracy
* Precision, Recall, Macro F1-score
* Cohen’s Kappa (agreement beyond chance)
* Cross-validation (model stability)
* Train-Test Gap (overfitting detection)
* Feature Ablation (robustness testing)

---

## 🔬 Key Findings

* All models achieved **>98% accuracy**, indicating strong predictive features
* **SVM and Random Forest** showed stable performance with minimal overfitting
* **MLR achieved high scores** due to indirect prediction approach
* Feature ablation revealed:

  * Models do not rely on a single feature
  * Pollutants exhibit correlation and redundancy
  * Removing engineered features (e.g., PM2.5/PM10 ratio) improved performance

---

## 🏆 Final Model Selection

The **Support Vector Machine (SVM)** was selected as the final model because:

* Directly solves the classification problem
* High accuracy and Macro F1-score
* High Cohen’s Kappa (~0.97)
* Minimal overfitting (low train-test gap)
* Strong robustness under feature ablation

---

## 📊 Visualizations

* Model comparison charts (Accuracy, F1-score, Kappa)
* Confusion matrix (best model)
* PCA-based decision boundary visualization (SVM)
* Feature ablation comparison plots

---

## 🔍 Insights

* Accuracy alone is insufficient → multi-metric evaluation is essential
* Feature importance ≠ feature dependency
* Robust models maintain performance even after feature removal
* Direct classification models outperform indirect regression approaches

---

## 🚀 Future Work

* Integrate meteorological data (temperature, humidity, wind speed)
* Perform advanced hyperparameter tuning (GridSearchCV)
* Explore deep learning approaches
* Deploy as a real-time AQI prediction system

---

## 🧠 Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib / Seaborn

---

## 📁 Project Structure

```
├── data/              # Dataset and documentation
├── notebooks/         # EDA, experiments, visualization
├── src/               # Model implementations
├── results/           # Outputs and plots
└── README.md
```

---

## 👨‍💻 Contributors

### 👤 Farhan Akhlaq

* Implemented **SVM model** and evaluation pipeline
* Handled **class imbalance (merging rare classes)**
* Built **PCA-based visualization with decision boundary**
* Led model comparison and final model selection

---

### 👤 Karthik

* Performed **Exploratory Data Analysis (EDA)**
* Conducted:

  * Correlation analysis
  * Skewness & kurtosis evaluation
* Engineered **seasonal feature**
* Identified key pollutant patterns and class imbalance
* Assisted in **Random Forest implementation and interpretation**

---

### 👤 Misbah

* Developed **Random Forest model**
* Performed:

  * Model training and evaluation
  * Cross-validation
  * Confusion matrix analysis
* Conducted **feature importance analysis**
* Interpreted environmental impact of pollutants

---

### 👤 Mukund

* Added and documented **AQI dataset (2015–2023)** 
* Performed **non-graphical EDA** including:

  * VIF-based multicollinearity analysis
  * Physical constraint validation
  * Class balance evaluation
* Built **Multiple Linear Regression (MLR)** pipeline
* Developed **prediction workflow and CLI-style interface**
* Implemented **feature ablation study with seasonal weighting**
* Maintained repository hygiene and documentation

---

## 📌 Conclusion

This project demonstrates how machine learning can be effectively applied to environmental data for AQI prediction. By combining **strong preprocessing, robust modeling, and comprehensive evaluation**, we developed a reliable and interpretable system for air quality classification.

---

## ⭐ Acknowledgment

This project was developed as part of a **machine learning mini-project**, focusing on real-world data analysis, model robustness, and evaluation best practices.
