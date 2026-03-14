# 📊 Dynamic Pricing System for Short-Term Rentals


[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)  
[![Pandas](https://img.shields.io/badge/pandas-1.6-green?logo=pandas)](https://pandas.pydata.org/)  
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2-orange?logo=scikit-learn)](https://scikit-learn.org/)  
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your-colab-link)  
[![View Notebook](https://img.shields.io/badge/nbviewer-view-orange)](https://nbviewer.org/github/yourusername/dynamic-pricing-system/blob/main/your_notebook.ipynb)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()  
[![Coverage](https://img.shields.io/badge/coverage-NA-lightgrey)]()  


---

## 🚀 Project Overview

The **Dynamic Pricing System** is a machine learning project designed to provide **data-driven nightly price recommendations** for short-term rental properties such as Airbnb listings.  

Setting the right price is a challenge for property hosts, they end up with pricing too high that it leads to fewer bookings, while pricing too low reduces their revenue. This system will leverage real historical data and property features from airbnb's across USA to **estimate optimal prices**, which can help the hosts to make informed, revenue-maximizing decisions.  

This project is currently **under development**, with ongoing improvements to the feature engineering, modeling, and evaluation pipeline.

---

## 🎯 Problem Statement

Hosts often set prices manually, without fully accounting for:

- Seasonal demand fluctuations  
- Market rates of nearby listings  
- Property amenities and features  
- Review scores and availability patterns  

This results in **suboptimal pricing decisions**.  

This project introduces a **dynamic, data-driven model** that learns from historical data and predicts **recommended prices** based on multiple property characteristics.

---

## 🧩 Features & Data

The model considers a variety of property and host features, including:

| Feature Category | Description |
|-----------------|-------------|
| Location | City, neighbourhood (ZIP code target-encoded) |
| Property | Room type, number of bedrooms, bathrooms, beds |
| Host | Verified status, total listings, experience as host |
| Availability | Minimum/maximum nights, annual availability |
| Reviews | Number of reviews, review scores |
| Derived Features | Target encoding for high-cardinality categorical features, one-hot encoding for low-cardinality features |

> High-cardinality features such as neighbourhood are **target-encoded** to represent average market value, preventing dimensionality explosion.

---

## 🛠️ Tools & Technologies

- **Python 3.11**  
- **Pandas / NumPy** – Data manipulation and cleaning  
- **Matplotlib / Seaborn** – EDA and visualization  
- **Scikit-learn** – Machine learning models (Linear Regression, Random Forest)  
- **Feature Engineering** – Scaling, one-hot encoding, target encoding  
- **Jupyter Notebook** – Interactive development and experimentation  

---

## 📈 Exploratory Data Analysis (EDA)

Key analyses performed:

- Histograms and boxplots for numeric features → identify skewness and outliers  
- Correlation heatmaps to inspect relationships with target (`price`)  
- Feature distributions across cities, room types, and neighbourhoods  
- Outlier detection and log transformations considered (for skewed numeric features)  

---

## 🔧 Machine Learning Pipeline

1. **Data Cleaning**  
   - Missing value imputation  
   - Dropping irrelevant or redundant columns  

2. **Feature Engineering**  
   - Target encoding for high-cardinality categorical variables  
   - One-hot encoding for low-cardinality categorical features (city, room_type)  
   - Scaling numeric features (for linear models)  

3. **Train/Test Split**  
   - Maintain separation for unbiased evaluation  

4. **Model Training & Evaluation**  
   - Linear Regression (baseline, scaled features)  
   - Random Forest Regressor (unscaled features)  
   - Metrics tracked: RMSE, MAE, R²  

> The pipeline is modular to allow easy integration of **additional models** and **advanced ensemble techniques** in future iterations.

---

## 📊 Potential Impact

This system can help hosts:

- Set **competitive prices**  
- **Maximize occupancy rates**  
- **Increase revenue**  
- Make pricing decisions **based on data rather than intuition**

It demonstrates the **real-world application of ML to dynamic pricing problems**, similar to how Airbnb, Uber, and Amazon adjust prices in real time.

---

## ⚡ Project Status

- [x] Dataset collection & cleaning (New York City Airbnb listings)  
- [x] Exploratory data analysis (EDA)  
- [x] Feature engineering & encoding  
- [x] Train/test split & scaling preparation  
- [ ] Model development & hyperparameter tuning **(in progress)**  
- [ ] Advanced ensemble & boosting methods **planned**  
- [ ] Deployment pipeline **planned for portfolio demonstration**  

> The model is actively being developed — further improvements will focus on predictive accuracy, feature optimization, and handling high-cardinality features more robustly.

---

## 📁 Dataset

The dataset used for this project was obtained from **[Kaggle – NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)**.  

It includes **listing details, host information, availability, reviews, and property characteristics**.

---

## 📌 How to Use

1. Clone the repository:

```bash
git clone https://github.com/AryanSharma1017/Dynamic-Pricing-System.git
cd Dynamic-Pricing-System
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

**will be provided when the project is completed**

3. Use Jupyter Notebook on Anaconda Navigator or VScode after setting up the python kernel

4. Run cells to explore data, preprocess, and train baseline models.



📝 Future Enhancements
* Will implement smoothed target encoding for neighbourhoods with few listings
* Hyperparameter tuning using GridSearchCV / RandomizedSearchCV will be used
* Use of stremalit to deploy this as a web-based interactive pricing tool
* Incorporation of seasonality and calendar trends
* Evaluate advanced ensemble models (XGBoost, LightGBM, CatBoost)


📫 Author 
Aryan Sharma
- [GitHub](https://github.com/AryanSharma1017)
- [LinkedIn](linkedin.com/in/aryansharma007)