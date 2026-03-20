# 📊 Dynamic Pricing System for Short-Term Rentals

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-1.6-green?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red)]()
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vLkKEMIrde4f2S5OQzjmz1PCgiDAidA7)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

---

## 🚀 Project Overview

The **Dynamic Pricing System** is a machine learning project designed to generate **data-driven nightly price recommendations** for short-term rental listings (e.g., Airbnb).

Pricing rental properties effectively is a complex problem — setting prices too high reduces bookings, while pricing too low leads to revenue loss. This system leverages historical listing data and advanced feature engineering techniques to **predict optimal pricing strategies**.

> 💡 This project demonstrates an end-to-end ML pipeline — from data preprocessing to model tuning and evaluation — with a focus on real-world applicability.

---

### Check it Out here 

Dynamic-Pricing-System: https://dynamic-pricing-system-bbovjcwe6xyf5ujvnmkdao.streamlit.app 


## 🎯 Problem Statement

Hosts often rely on intuition rather than data, ignoring key factors such as:

* Local market trends
* Property characteristics
* Listing availability
* Review activity
* Competition within neighbourhoods

This results in **suboptimal pricing decisions**.

👉 This project builds a **predictive pricing engine** that learns patterns from historical data to recommend competitive prices.

---

## 🧠 Key Highlights

* ✔️ End-to-end ML pipeline (EDA → Feature Engineering → Modeling → Evaluation)
* ✔️ Advanced feature engineering (log transforms, target encoding, clustering)
* ✔️ Handling of high-cardinality features (neighbourhood encoding)
* ✔️ Hyperparameter tuning using RandomizedSearchCV
* ✔️ Comparison of multiple models (Linear, KNN, Random Forest, Boosting)
* ✔️ Real-world dataset with 220K+ listings

---

## 🧩 Features & Engineering

### 🔹 Core Features

* **Location** → City, Neighbourhood (target encoded)
* **Property** → Room type
* **Availability** → Minimum nights, availability_365
* **Reviews** → Number of reviews
* **Host** → Listing count

---

### 🔹 Advanced Feature Engineering

* 📌 **Log Transformations** → Reduced skewness in numerical features
* 📌 **Target Encoding** → Neighbourhood pricing signal
* 📌 **KMeans Clustering (Location Intelligence)** → Grouped latitude/longitude into meaningful regions
* 📌 **One-Hot Encoding** → Low-cardinality features (city, room type, clusters)
* 📌 **Outlier Handling** → Removed extreme pricing and minimum night anomalies

> 🚀 These steps significantly improved model performance and stability.

---

## 🛠️ Tech Stack

* **Python 3.11**
* **Pandas / NumPy** – Data processing
* **Matplotlib / Seaborn** – Visualization
* **Scikit-learn** – ML models & pipelines
* **XGBoost** – Gradient boosting model
* **Jupyter Notebook / Google Colab** – Development

---

## 📈 Model Performance

After feature engineering and hyperparameter tuning:

| Model             | MAE   | RMSE  | R² Score  |
| ----------------- | ----- | ----- | --------- |
| Random Forest ✅   | 0.346 | 0.466 | **0.595** |
| XGBoost           | 0.360 | 0.478 | 0.574     |
| Gradient Boosting | 0.367 | 0.485 | 0.561     |


Also tested out other regression models with the scaled dataset 

| Model             | MAE   | RMSE  | R² Score  |
| ----------------- | ----- | ----- | --------- |
| Linear Regression  | 0.73 | 169.3  | 0.22      |
| KNN.              | 0.670 | 142.67 | 0.344     |



> 📊 The Random Forest model achieved the best performance, explaining ~59% of price variance.

---

## 🔧 Machine Learning Pipeline

1. **Data Cleaning**

   * Removed invalid and extreme values
   * Handled missing data

2. **Feature Engineering**

   * Log transformation of skewed features
   * Target encoding for neighbourhood
   * KMeans clustering on geolocation
   * One-hot encoding for categorical variables

3. **Train/Test Split**

   * Strict separation to prevent data leakage

4. **Scaling**

   * Applied where required (linear models)

5. **Model Training**

   * Linear Regression (baseline)
   * KNN
   * Random Forest (best performer)
   * Gradient Boosting
   * XGBoost

6. **Hyperparameter Tuning**

   * RandomizedSearchCV for optimal performance

---

## 📊 Exploratory Data Analysis (EDA)

* Distribution analysis of pricing and features
* Detection of extreme outliers
* Correlation analysis with target variable
* Feature behavior across cities and room types

---

## 💡 Key Insights

* 📍 Location is the strongest pricing driver
* 🏠 Room type significantly impacts price
* 📉 Extreme values (outliers) degrade model performance
* 🔄 Log transformation improves model stability
* 🤖 Ensemble models outperform linear approaches

---

## 📌 Dataset

* Source: [Kaggle – NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
* Size: **226,000+ listings**
* Features: Location, availability, reviews, host info

---

## ⚡ Project Status

* [x] Data cleaning & preprocessing
* [x] Feature engineering
* [x] Model training & evaluation
* [x] Hyperparameter tuning
* [ ] Model stacking (planned)
* [ ] Deployment (Streamlit)
* [ ] Advanced feature engineering (ongoing)

---

## 🚀 Future Improvements

* 🔹 Model stacking / ensembling
* 🔹 Advanced location features (distance-based metrics)
* 🔹 Time-based features (seasonality)
* 🔹 Deployment using Streamlit
* 🔹 Integration with real-time pricing APIs

---

## 🧑‍💻 How to Run

```bash
git clone https://github.com/AryanSharma1017/Dynamic-Pricing-System.git
cd Dynamic-Pricing-System
pip install -r requirements.txt
```

Run the notebook in Jupyter or Colab.

---

## 📫 Author

**Aryan Sharma**

* GitHub: https://github.com/AryanSharma1017
* LinkedIn: https://www.linkedin.com/in/aryansharma007

---

## ⭐ Final Note

This project demonstrates not just model building, but **practical machine learning skills** including:

* Feature engineering
* Handling real-world messy data
* Model tuning and evaluation
* Avoiding data leakage

> 🚀 Built with the mindset of solving real-world pricing problems using machine learning.
