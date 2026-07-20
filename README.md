# 🌫️ Air Quality Index (AQI) Classification using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview

This project focuses on predicting **Air Quality Index (AQI) categories** using supervised machine learning models. The prediction is based on pollutant concentrations and contextual features such as **city** and **season**.

The objective was to build **accurate, robust, and interpretable models**, while performing comprehensive evaluation using multiple metrics and validation techniques.

---

## 📊 Dataset
* **Source:** [India City Air Quality Index Dataset (2015-2023) - Kaggle](https://www.kaggle.com/datasets/tushardobal/india-city-air-quality-index-dataset-20152023)

The dataset consists of air pollution measurements collected across multiple Indian cities (2015–2023), including:

### 🔹 Features
* **Pollutants:** PM2.5, PM10, NO₂, SO₂, CO, O₃
* **Contextual:** City, Season (engineered from date)

### 🔹 Target
* **AQI Category:** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)
* *Note: Target features were engineered specifically to perform seasonal analysis.*

### 🔹 Preprocessing Steps
1. Handling missing values and validating imputation integrity.
2. Feature selection and consistency checks.
3. One-hot encoding of categorical variables.
4. Feature scaling (using `StandardScaler` for SVM).
5. Validation of physical constraints (e.g., ensuring PM2.5 ≤ PM10).
6. Dataset standardization and cleaning.

---

## ⚙️ Models Implemented

### 1️⃣ Support Vector Machine (SVM)
* **Kernel:** Radial Basis Function (RBF)
* **Hyperparameters:** `C = 10`, `gamma = 'scale'`
* **Features:** Applied on standardized features.
* **Results:** Achieved strong generalization and stable performance.

---

### 2️⃣ Random Forest
* **Type:** Ensemble-based decision tree classifier.
* **Characteristics:** Low variance and strong stability.
* **Usage:** Used extensively for feature importance analysis.

---

### 3️⃣ Multiple Linear Regression (MLR)
* **Indirect Approach:**
  1. Predict numerical AQI value.
  2. Map predictions to categorical AQI bands.
* **Results:** High numerical performance, but less suitable for direct classification.

---

## 📈 Evaluation Metrics

To ensure a fair and reliable evaluation, the following metrics were tracked:
* **Accuracy**
* **Precision, Recall, and Macro F1-score**
* **Cohen’s Kappa** (agreement beyond chance)
* **K-Fold Cross-Validation** (model stability)
* **Train-Test Gap** (overfitting detection)
* **Feature Ablation** (robustness testing)

---

## 🔬 Key Findings

* **High Performance:** All models achieved **>98% accuracy**, indicating highly predictive features in the dataset.
* **Stability:** **SVM and Random Forest** showed highly stable performance with minimal overfitting.
* **Indirect Performance:** **MLR achieved high scores** but is fundamentally a regression approach adapted for classification.
* **Feature Ablation Study:**
  * Models do not rely on a single feature.
  * Pollutants exhibit high correlation and redundancy.
  * Removing engineered features (e.g., PM2.5/PM10 ratio) actually improved overall performance.

---

## 🏆 Final Model Selection

The **Support Vector Machine (SVM)** was selected as the final model due to:
* Directly solving the multi-class classification problem.
* Leading accuracy and Macro F1-score.
* High Cohen’s Kappa (~0.97).
* Minimal overfitting (lowest train-test gap).
* Strong robustness under feature ablation.

---

## 📊 Visualizations & Reports

The project results and visualizations are documented in the following folders:
* Model comparison charts (Accuracy, F1-score, Kappa)
* Confusion matrix (best model)
* PCA-based decision boundary visualization (SVM)
* Feature ablation comparison plots

### 📄 Project Documents
You can access the project report and presentation slides directly on GitHub:
* [Project Report (PDF)](https://github.com/MukundXplore/Seasonal-AQI-Analysis-Through-Classical-ML-Models/blob/main/report/Seasonal%20Air%20Quality%20Index%20(AQI)%20Category%20Prediction%20Using%20Classical%20Supervised%20Machine%20Learning%20Models.pdf)
* [Presentation (PDF)](https://github.com/MukundXplore/Seasonal-AQI-Analysis-Through-Classical-ML-Models/blob/main/report/PPTX%20Seasonal%20Air%20Quality%20Index%20(AQI)%20Category%20Prediction%20Using%20Classical%20Supervised%20Machine%20Learning%20Models.pdf)
* [Presentation Slides (PPTX)](https://github.com/MukundXplore/Seasonal-AQI-Analysis-Through-Classical-ML-Models/blob/main/report/Seasonal%20Air%20Quality%20Index%20(AQI)%20Category%20Prediction%20Using%20Classical%20Supervised%20Machine%20Learning%20Models.pptx)

---

## 🔍 Key Insights

1. **Beyond Accuracy:** Accuracy alone is insufficient; multi-metric evaluation (F1-score, Kappa) is essential.
2. **Feature Interdependence:** Feature importance does not equate to feature dependency.
3. **Robustness:** Robust models maintain performance even after feature removal.
4. **Methodology:** Direct classification models outperform indirect regression-to-classification approaches.

---

## 🚀 Future Work

* Integrate meteorological data (temperature, humidity, wind speed).
* Perform advanced hyperparameter tuning via `GridSearchCV`.
* Explore deep learning architectures (e.g., Multilayer Perceptrons).
* Deploy as a real-time web interface for AQI prediction.

---

## 🧠 Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure

```text
.
├── aqi-project/
│   ├── data/          # Dataset and related documentation
│   ├── notebooks/     # Jupyter Notebooks for EDA, SVM, RF, and MLR models
│   └── src/           # Implementation source scripts
├── report/            # PPTX presentation and PDF report files
├── .gitignore         # Version control ignore list
└── README.md          # Project overview and documentation
```

---

## 👨‍💻 Contributors

### 👤 Farhan Akhlaq
* Implemented **SVM model** and evaluation pipeline.
* Handled **class imbalance** (merging rare classes).
* Built **PCA-based visualization** with decision boundary.
* Led model comparison and final model selection.

---

### 👤 Karthik Baurai
* Performed **Exploratory Data Analysis (EDA)**.
* Conducted correlation analysis and skewness/kurtosis evaluation.
* Engineered the **seasonal feature**.
* Identified key pollutant patterns and class imbalance.
* Assisted in **Random Forest implementation and interpretation**.

---

### 👤 Misbah Ul Islam
* Developed the **Random Forest model**.
* Performed model training, evaluation, and cross-validation.
* Conducted **confusion matrix and feature importance analysis**.
* Interpreted environmental impact of pollutants.

---

### 👤 Mukund Prasad
* Added and documented the **AQI dataset (2015–2023)**.
* Performed **non-graphical EDA** (VIF multicollinearity, physical constraints).
* Built the **Multiple Linear Regression (MLR)** pipeline.
* Developed prediction workflow and CLI-style interface.
* Implemented the **feature ablation study with seasonal weighting**.
* Maintained repository hygiene and documentation.

---

## 📌 Conclusion

This project demonstrates how machine learning can be effectively applied to environmental data for AQI prediction. By combining **strong preprocessing, robust modeling, and comprehensive evaluation**, we developed a reliable and interpretable system for air quality classification.

---

## 🎓 Academic Context & Guidance

This project was completed as part of the **Machine Learning** course requirement during the **4th Semester**:

* **Course Title:** Machine Learning
* **Semester:** 4th Semester
* **Department:** Department of Computer Science & Engineering
* **Faculty:** Faculty of Engineering & Technology
* **Institution:** South Asian University (SAU), New Delhi

### 👨‍🏫 Course Instructor & Advisor
* **Prof. Ashwini B.**  
  [LinkedIn Profile](https://www.linkedin.com/in/ashwini-bhamini/) | Department of Computer Science & Engineering, South Asian University

---

## ⭐ Acknowledgment

This project was developed as part of a **Machine Learning project**, focusing on real-world data analysis, model robustness, and evaluation best practices.
