# REIT ETFs Forecasting
Author: [JingYun (Jonna) Wang](/jingyunwang24@gmail.com)

🌻 Currently working on it. 🌻

## Overview
This project explored on how to use machine learning modeling to predict future price...

## Business Problem
...

## Data
The historical price data from [Yahoo Finance](https://finance.yahoo.com/) are free to acquire through [yfinance API](https://pypi.org/project/yfinance/). Here, I focused on <b>REIT ETFs</b> (Real Estate Investment Trust Exchange-Traded Funds). [Top 28 REIT ETFs](https://etfdb.com/etfdb-category/real-estate/) ticker names are selected for further analysis. The last date for all ETFs is 06/23/2021. 

## Methods
A variety of preprocessing steps was applied to different modeling technique. For regression 
Preprocessing steps include....


### Regression: forecast closing price one month ahead
Three different modeling techniques were used to predict the ETF prices.</br>
1. [ARIMA](./notebook/02_ARIMA.ipynb)
2. [FBProphet](./notebook/03_FBProphet.ipynb)
3. [LSTM](./notebook/04_LSTM.ipynb)

### Classification: forecast one month ahead
Scikit-learn enabled method 

## Model Evaluation


## Results

### Regression
|  Model | MAE | RMSE | R^2 | Choice |
| :---: | :---: |:---: | :---: | :---: |
| ARIMA | 4.7131 | 5.0823 | -5.688| 
| FBProphet | 14.0614 | 14.1760 | -51.0399 | 
| LSTM | 0.7419 | 0.9534 | 0.7756| ✓ |

### Classification
|  Model | Accuracy | ROC-AUC | Choice |
| :---: | :---: |:---: | :---: |
| Logistic Regression | 
| KNN |  
| Random Forest |
| Bagging |
| XGBoost |
| AdaBoost |
| Gradient Boosting |
| SVC |
| NuSVC |





## Conclusion


## Future Work
Further analysis can be explored on the following to provide additional insights and improve the model performance.
- Recommend trading decisions => profit calculator
- Consider exogenous variables
- Model deployment

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