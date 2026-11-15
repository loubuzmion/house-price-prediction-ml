# house-price-prediction-ml
# House Price Prediction: Linear Regression vs Random Forest

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

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## Files
- `house_price_prediction.ipynb`: full analysis and modelling
- `house_sales_prediction.csv`: dataset
