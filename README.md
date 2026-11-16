# Data Analyst Portfolio:
# House Price Prediction: Linear Regression vs Random Forest

Technical Skills: Python

Soft Skills: 

## Education:
* Diploma in Digital Technology Development and Design (Level 5)
  Data Analyst

## Project: Predicting House Price with Multiple Linear Regression
Overview
* Objective: Build a Multiple Linear Regression (MLR) model to predict house prices (Price) using various physical and locational features.
  
* Dataset Features:
  - Square_Feet: The total area of the house in square feet.
  - Num_Bedrooms: The number of bedrooms in the house.
  - Num_Bathrooms: The number of bathrooms in the house.
  - Num_Floors: The number of floors in the house.
  - Garage_Size: The size of the garage.
  - Location_Score: A score representing the desirability of the location.
  - Distance_to_Center: The distance (in km) to the city center.
  - Price: The price of the house (target variable). 

Step 2: Exploratory Data Analysis (EDA)
1. Summary Statistics
- Using df_house_price.describe() provides statistical metrics such as mean, standard deviation, minimum, and quartiles for all numerical columns.
- For instance, the average house price is approximately $\$582,209.63$ with a standard deviation of $\$122,273.39$, ranging from a minimum of $\$276,892.47$ to a maximum of $\$960,678.27$.   

2. Visualizations & Insights
- Square Feet vs. Price (Scatter Plot):
  - Demonstration: Visualizes the relationship between house size (Square_Feet) on the x-axis and price (Price) on the y-axis using Seaborn and Matplotlib with custom transparency (alpha=0.5).
  - Insight: There is a moderate positive relationship between house size and price, meaning larger houses generally cost more. However, the wide spread of points indicates that square footage alone does not entirely determine price.   

![Square Feet vs Price](square-feet-vs-price.png)

- Distribution of Price (Histogram with KDE):
  - Demonstration: Plots the frequency distribution of the target variable Price using sns.histplot with 30 bins and an overlaid Kernel Density Estimate (KDE) curve.
  - Insight: Price is roughly bell-shaped and fairly symmetric, centred at around $\$550\text{k}$ to $\$600\text{k}$. Prices range from about $\$277\text{k}$ to $\$961\text{k}$ with no extreme outliers, which suits a linear regression model.
 
![Square Feet vs Price](square-feet-vs-price.png)


























An end-to-end machine learning project that predicts house prices from property features such as size, bedrooms, bathrooms, floors, garage size, location score and distance to the city center.

## What I did
- Loaded and checked a 500-row housing dataset (no missing values)
- Ran exploratory data analysis: summary statistics, scatter plots, price histogram, pairplot and correlation heatmap
- Split the data 80/20 into train and test sets
- Trained a Multiple Linear Regression model and a Random Forest Regressor (200 trees)
- Compared both models using RMSE and actual-vs-predicted plots

## Results
| Model | RMSE |
|---|---|
| Multiple Linear Regression | $63,952 |
| Random Forest | $71,999 |

Linear Regression performed better on this dataset (test R² ≈ 0.73), which suggests the price relationships are mostly linear.

![Actual VS Predicted](actual-vs-predicted-plt.png)

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## Files
- `house_price_prediction.ipynb`: full analysis and modelling
- `house_sales_prediction.csv`: dataset
