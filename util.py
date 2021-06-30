# Utility file for Time Series Analysis

# 1 - visualize time series
# 2 - stationary check
# 3 - decompostion
# 4 - detrend method
# 5 - ACF & PACF
# 6 - evaluate

# General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# pmdarima
import pmdarima as pm
from pmdarima.arima import decompose
from pmdarima.utils import decomposed_plot
from pmdarima.arima.stationarity import ADFTest
from pmdarima.arima.utils import ndiffs
from pmdarima import model_selection

# statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA



# 1
def visualize_time_series(TS):
    '''
    TS: Time Series
    '''
    TS.plot(figsize=(15,9))
    plt.title("Close")
    plt.ylabel("Price")
    plt.show()



# 2
def stationary_check_statsmodels(TS, window_size):
    '''
    statsmodels
    
    TS: Time Series
    
    window_size: parameter for window
    
    '''
    
    # Rolling Statistics
    roll_mean = TS.rolling(window=window_size, center=False).mean()
    roll_std = TS.rolling(window=window_size, center=False).std()
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(TS, color='blue', label='Original')
    plt.plot(roll_mean, color = 'red', label='Rolling Mean')
    plt.plot(roll_std, color = 'black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title("Rolling Mean and Standard Deviation")
    plt.show(block=False)
    
    
    # Dickey-Fuller Test
    dftest = adfuller(TS)
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 
                                             'p-value', 
                                             '#Lags Used', 
                                             'Number of Observations Used'])
    
    # Determine stationarity based on p value
    if dfoutput[1] < 0.05:
        print("Stationary, because p < 0.05 \n")
    else:
        print("Non-stationary, because p ≥ 0.05 \n")
    
    # Print Dickey-Fuller Test Result
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        
    print('Results of Dickey-Fuller test: \n')
    print(dfoutput)
  
    
def stationary_check_pmdarima(TS):
    '''
    pmdarima
    TS: time series
    
    '''
    
    adf_test = ADFTest(alpha=0.05)
    p_val, should_diff = adf_test.should_diff(TS) 

    print(f"P-Value: {p_val}, so should you difference the data? {should_diff}")


    
# 3
def decomposition_plot_pmdarima(TS, frequency):
    '''
    pmdarima
    
    TS: Time Series
    
    frequency: parameter for m
    '''
    # Use decompose function from pmdarima
    decomposed = decompose(TS.values, 'multiplicative', m=frequency)
                                      # use "multiplicative" when see an increasing trend
    # Plot the decomposition plot
    decomposed_plot(decomposed, figure_kwargs={'figsize': (12,10)})
    
 
def decomposition_plot_statsmodels(TS, frequency):
    '''
    statsmodels
    
    TS: Time Series
    
    frequency: parameter for freq
    '''
    
    decomposition = seasonal_decompose(TS, freq=frequency)

    # Gather the trend, seasonality, and residuals 
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot gathered statistics
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(TS, label='Original', color='blue')
    plt.legend(loc='best')

    plt.subplot(412)
    plt.plot(trend, label='Trend', color='blue')
    plt.legend(loc='best')

    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality', color='blue')
    plt.legend(loc='best')

    plt.subplot(414)
    plt.plot(residual, label='Residuals', color='blue')
    plt.legend(loc='best')

    plt.tight_layout()



# 4
def detrend_transformation(TS, log=False, sqrt=False):
    '''
    Log Transformation
    
    Sqrt Transformation
    
    '''
    
    if log == True:
        TS_log = np.log(TS)
        return TS_log

    elif sqrt == True:
        TS_sqrt = np.sqrt(TS)
        return TS_sqrt
    
    else:
        print("Error")
        return None
        

def detrend_rolling_mean(TS, regular=False, window_size=None, half_life=None):
    '''
    Subtract rolling mean
    
    Subtranct exponential rolling mean
    
    '''
    
    if regular == True:
        rolling_mean = TS.rolling(window=window_size).mean()
            
        TS_minus_rolling_mean = TS - rolling_mean
        TS_minus_rolling_mean.dropna(inplace=True)
        return TS_minus_rolling_mean
    
    else:
        exp_rolling_mean = TS.ewm(halflife=half_life).mean()
        
        TS_minus_exp_rolling_mean = TS- exp_rolling_mean
        TS_minus_exp_rolling_mean.dropna(inplace=True)
        return TS_minus_exp_rolling_mean
    

def detrend_differencing(TS, periods):
    '''
    Just differencing method
    
    '''
    
    TS_diff = TS.diff(periods=periods)
    TS_diff.dropna(inplace=True)
    return TS_diff



# 5
def plot_ACF_PACF(TS):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,4))
    plot_acf(TS, ax=ax1, lags=25)
    plot_pacf(TS, ax=ax2, lags=25)

def pd_ACF(TS):
    pd.plotting.autocorrelation_plot(TS)



# 6
def evaluate(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    print("MSE: %.3f " % MSE)
    print("MAE: %.3f" % MAE)
    print("MAPE:%.3f " % MAPE)