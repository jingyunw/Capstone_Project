# REIT ETFs Forecasting
Author: [JingYun (Jonna) Wang](/jingyunwang24@gmail.com)

🌻 Currently working on it. 🌻

<img src="images/stock.jpeg">


## Overview
This project uses a regression model for forecasting future prices and a classification model to predict the trend for REIT ETFs. The historical data acquired from Yahoo Finance contains up-to-date price information. Potential investors can use my models to predict both future prices and trends.


## Business Problem
### Motivation
Invest in real estate over long term could offer a reliable stream of income. This could also offer the benefit of tax reduction, direct ownership, potential growth of net worth, and so on. But those benefits come with a lot of hands-on responsibilities. Invest in REIT ETFs (Real Estate Investment Trust Exchange-Traded Funds) offer a low investment requirement and hands-off (but less control) way to earn dividends quarterly by purchasing baskets of stocks.
***
### Goal
Predict future price and trend through machine learning.


## Data
The historical price data from [Yahoo Finance](https://finance.yahoo.com/) are free to acquire through [yfinance API](https://pypi.org/project/yfinance/). Here, I focused on [Top REIT ETFs](https://etfdb.com/etfdb-category/real-estate/) for further analysis. The last date for all ETFs was 06/23/2021.


## Methods
Regression and classification models were built on VNQ. Train-val-test split was performed to evaluate machine learning performance. Both models were applied to the rest of the 14 REIT ETFs to see how the model is generalizable for different time series.

### Regression: 
1. [ARIMA](./notebook/2_ARIMA.ipynb)
2. [FBProphet](./notebook/3_FBProphet.ipynb)
3. [LSTM](./notebook/4_LSTM.ipynb)

***

### Classification:
[Classifier](./notebook/5_Classification.ipynb)</br>
Logistic Regression, KNN, Random Forest, Bagging, XGBoost, AdaBoost, Gradient Boosting, SVC, NuSVC 


## Model Evaluation
Best Performance Model
|  Model | MAE | RMSE | R^2 | Choice |
| :---: | :---: |:---: | :---: | :---: | 
| LSTM | 0.7419 | 0.9534 | 0.7756| ✓ |


|  Model | Accuracy | F1(0) | F1(1) | Choice |
| :---: | :---: |:---: | :---: |:---: |
| NuSVC | 0.9091 | 0.90 | 0.92 | ✓ |


## Results
...







## Conclusion
By analyzing VNQ time series, the LSTM model was selected for regression and the NuSVC model was selected for classification. Both models were applied to the rest of the 14 REIT ETFs to see how the model is generalizable for different time series. LSTM is a good fit for price prediction no matter the time series has either an increasing trend or a decreasing trend. NuSVC is more like to fit for time series which have a closer price range with VNQ.

***
### Recommendation
- <b>Update</b>: Update the model periodically so that the model can learn new pattern


## Future Work
Further analysis can be explored on the following to provide additional insights and improve the model performance.
- <b>Buy/Hold/Sell</b>: Create an alert when the prediction reaches certain point
- <b>Market Sentiment</b>: Web scrapping on social media to gather market information 
- <b>Extensive application</b>: Further develop models for other investment products with time series based
- <b>Model Deployment</b>: Automaticaly fetch new historical data, and run models for prediction


## For More Information
See the full analysis and modeling in the [Jupyter Notebook](./Notebook.ipynb) and [presentation](./Presentation.pdf).
For additional information please contact, JingYun (Jonna) Wang at jingyunwang24@gmail.com

## Repository Structure
```
├── data
├── images
├── models
├── notebook
│   ├── 1_Fetch_Data.ipynb
│   ├── 2_ARIMA.ipynb
│   ├── 3_FBProphet.ipynb
│   ├── 4_LSTM.ipynb
│   └── 5_Classification.ipynb
├── Notebook.ipynb
├── Presentation.pdf
├── README.md
└── util.py
```