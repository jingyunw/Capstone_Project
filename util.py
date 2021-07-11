# Utility file for Time Series Analysis #


## Preprocessing ##
# - set the "Date" as the index column for a df

## Sklearn Evaluation ##
# - MAE, RMSE, R^2

## ARIMA ##
# 1 - visualize time series
# 2 - stationary check
# 3 - decompostion
# 4 - detrend method
# 5 - ACF & PACF

## LSTM ##
# 1 - LSTM model evaluation
# 2 - plot true vs. prediction

## Classification ##
# 1 - evaluate classification model
# 2 - comparison df for different classification model





###############
#   Imports   #
###############

# General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score                                            # regression
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix, roc_auc_score, plot_roc_curve  # classification

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



#####################
#   Preprocessing   #
#####################

def preprocess_df(df):
    '''
    Preprocess the dataframe by changing the "Date" column to datetime64 and set as index. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Input:
    - df: Panda dataframe
    
    Output:
    - Panda dataframe with date as index
    '''

    # Convert Date column to datetime64 type
    df['Date'] = pd.to_datetime(df['Date'])

    # Set index to a datetime index
    df.set_index('Date', inplace=True)
    print("This dataframe's index is in datatime64?", df.index.inferred_type == "datetime64")
    # Check
    display(df)




#########################
#   Sklearn Evalution   #
#########################


def evaluate(y_true, y_pred):
    '''
    Evaluate MAE, RMSE, R^2 between true y and predicted y.

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - y_true, y_pred: true y and predicted y
    
    Outputs:
    - Evalution results for true y and predicted y
        - MAE, RMSE, and R^2
    '''

    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    R2 = r2_score(y_true, y_pred)
    
    print("MAE: %.4f" % MAE)
    print("RMSE: %.4f" % RMSE)
    print("R^2: %.4f" % R2)






#############
#   ARIMA   #
#############

# 1
def visualize_time_series(TS):
    '''
    Visualize a time series. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Input:
    - TS: Time series
    
    Output:
    - Time series plot
    '''

    TS.plot(figsize=(15,9))
    plt.title("Close")
    plt.ylabel("Price")
    plt.show()


# 2a
def stationary_check_statsmodels(TS, window_size):
    '''
    Stationary check using statsmodels library. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - TS: Time series
    - window_size: integer
    
    Outputs:
    - Plot of the original time series, rolling mean, rolling std
    - Dickey-Fuller test statistic with p value shown
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
  
# 2b    
def stationary_check_pmdarima(TS):
    '''
    Stationary check using pdmarima library. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Input:
    - TS: Time series
    
    Output:
    - Dickey-Fuller test statistic with p value shown
    - Whether or not shoud differencing
    '''
    
    adf_test = ADFTest(alpha=0.05)
    p_val, should_diff = adf_test.should_diff(TS) 

    print(f"P-Value: {p_val}, so should you difference the data? {should_diff}")

    
# 3a
def decomposition_plot_pmdarima(TS, frequency):
    '''
    Decomposition plot using pdmarima library. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - TS: Time series
    - frequency: integer
    
    Outputs:
    - Decomposition plot showing the original time series, trend, seasonality, and residuals
    '''
    
    # Use decompose function from pmdarima
    decomposed = decompose(TS.values, 'multiplicative', m=frequency)
                                      # use "multiplicative" when see an increasing trend
    # Plot the decomposition plot
    decomposed_plot(decomposed, figure_kwargs={'figsize': (12,10)})
    
# 3b 
def decomposition_plot_statsmodels(TS, frequency):
    '''
    Decomposition plot using statsmodels library. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - TS: Time series
    - frequency: integer
    
    Outputs:
    - Decomposition plot showing the original time series, trend, seasonality, and residuals
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


# 4a
def detrend_transformation(TS, log=False, sqrt=False):
    '''
    Detrend time series data by using either log or square root transformation. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - TS: Time series
    - log: True for performing log transformtion
    - sqrt: True for performing square root transformation
    
    Return:
    - Detrended time series
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
        
# 4b
def detrend_rolling_mean(TS, regular=False, window_size=None, half_life=None):
    '''
    Detrend time series by subtracting either the rolling mean or the weighted rolling mean. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - TS: Time series
    - regular: True for using .rolling(), False for using .ewm()
    - window_size: integer (applied if subtract rolling mean)
    - half_life: integer (applied if subtract weighted rollingmean)

    Return:
    - Detrended time series
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
    
# 4c
def detrend_differencing(TS, periods):
    '''
    Detrend time series by differencing. 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - TS: Time series
    - periods: integer

    Return:
    - Detrended time series
    '''
    
    TS_diff = TS.diff(periods=periods)
    TS_diff.dropna(inplace=True)
    return TS_diff


# 5a
def plot_ACF_PACF(TS):
    '''
    Plot the autocorrelation and partial-autocorrelation of time series using statsmodels library.

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - TS: Time series

    Outputs:
    - ACF and PACF
    '''

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,4))
    plot_acf(TS, ax=ax1, lags=25)
    plot_pacf(TS, ax=ax2, lags=25)

# 5b
def pd_ACF(TS):
    '''
    Plot the autocorrelation of time series using panda library 

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - TS: Time series

    Outputs:
    - ACF and PACF
    '''
    
    pd.plotting.autocorrelation_plot(TS)





##############
#   LSTM   ###
##############

# 1
def lstm_model_evaluation(model, X_train, y_train, X_val, y_val):
    '''
    Evaluate the LSTM model by using  prediction of y based on X.
    Inverse transform for both the prediction y and actual y. 
    
    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Inputs:
    - model: Time series model name
    - X_train, y_train, X_val, y_val
    
    Outpus:
    - Evalution results for true y and predicted y
        - MAE, RMSE, and R^2
    
    Return:
    - y_val_true, y_val_inv

    '''
    
    # Make prediction
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Inverse transform the prediction and actual
    y_train_inv = scaler.inverse_transform(y_train_pred) # prediction
    y_val_inv = scaler.inverse_transform(y_val_pred) # prediction

    y_train_true = scaler.inverse_transform(y_train) # actual
    y_val_true =scaler.inverse_transform(y_val) # actual
    
    # Use the evalute function from util pyfile
    print("Train results: ")
    print(ut.evaluate(y_train_true, y_train_inv))
    print("\n")
    print("Val results: ")
    print(ut.evaluate(y_val_true, y_val_inv))

    return y_val_true, y_val_inv


# 2
def lstm_plot_prediction(y_true, y_pred):
    '''
    Comparison plot of actual y and predicted y for LSTM model.

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - y_true, y_pred: true y and predicted y
    
    Outpus:
    - Scatter plot of true y and line plot of predicted y
    '''

    plt.figure(figsize=(20,10))
    plt.plot(y_true, '.')
    plt.plot(y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.show()





#######################
#   Classification    #
#######################

# 1
def class_model_evaluation(model, X_train, y_train, X_test, y_test, use_decision_function='yes'):
    '''
    Evaluate a classfication model in terms of accuracy, and roc-auc-score.
    Plot confusion matrix and roc-curve for test set.

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - model: classification model name
    - X_train, y_train, X_test,  y_test
    - use_decision_function: yes, no, skip

    Outputs:
    - Train/Test accuracy, roc-auc score 
    - classification report, confusion matrix, roc-curve

    Return:
    - train_acc, test_acc, train_roc_auc, test_roc_auc
    '''
    
    # accuracy
    train_acc = []
    test_acc = []
    
    # roc-auc score
    train_roc_auc = []
    test_roc_auc = []


    # Prediction
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities
    if use_decision_function == 'skip': # skips calculating the roc_auc_score
        train_score = False
        test_score = False
    
    elif use_decision_function == 'yes': # not all classifiers have decision_function
        train_score = model.decision_function(X_train)
        test_score = model.decision_function(X_test)
    
    elif use_decision_function == 'no':
        train_score = model.predict_proba(X_train)[:, 1] # proba for the 1 class
        test_score = model.predict_proba(X_test)[:, 1]
    
    else:
        raise Exception ("The value for use_decision_function should be 'skip', 'yes' or 'no'.")
    

    # Train
    print("Train")
    print("-*-*-*-*-*-*-*-*")
    print(f"accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    train_acc.append(round(accuracy_score(y_train, y_train_pred),4))
    
    if type(train_score) == np.ndarray:
        print(f"roc-auc: {roc_auc_score(y_train, train_score):.4f}", "\n")
    train_roc_auc.append(round(roc_auc_score(y_train, train_score), 4))


    # Test
    print("Test")
    print("-*-*-*-*-*-*-*-*")
    print(f"accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    test_acc.append(round(accuracy_score(y_test, y_test_pred), 4))
    
    if type(test_score) == np.ndarray:
        print(f"roc-auc: {roc_auc_score(y_test, test_score):.4f}")
    test_roc_auc.append(round(roc_auc_score(y_test, test_score), 4))
    
    print("\n")


    # Classification Report
    print(classification_report(y_test, y_test_pred))

    # Confusion Matrix
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, values_format = '.0f')

    # Plot ROC-curve
    plot_roc_curve(model, X_test, y_test)

    plt.show()

    return train_acc, test_acc, train_roc_auc, test_roc_auc



# 2
def class_model_comparison(model_results):
    '''
    Create a df to compare the different classification model reults based on 
    train/test accuracy and roc-auc score

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Input:
    - model_results: list contains different classification result

    Return:
    - model comparsion df

    '''
    
    new_model_results = []

    # Change the tuple list to 1D list for each model and put the results in the new list
    for result in model_results:
        result = sum(result[:], []) # Change to 1D list
        new_model_results.append(result)
    

    # Create a new df
    model_comp_df = pd.DataFrame(columns=['train_acc', 'test_acc', 'acc_diff', 
                                         'train_roc_auc', 'test_roc_auc', 'roc_auc_diff'],
                                index=['Logistic Regression', 'KNN', 
                                       'Random Forest', 'Bagging',
                                       'XGBoost', 'AdaBoost', 'GradientBoost', 
                                       'SVC', 'NuSVC'])
    
    # Append the new_models_results to the corresponding position in the df
    for i in range(9): # total of 9 classifiers
        # For each inner list, the...
        # 1st element: train_acc
        model_comp_df['train_acc'][i] = new_model_results[i][0]
        
        # 2nd element: test_acc
        model_comp_df['test_acc'][i] = new_model_results[i][1]
        
        # 3rd element: train_roc_auc
        model_comp_df['train_roc_auc'][i] = new_model_results[i][2]
        
        # 4th element: test_roc_auc
        model_comp_df['test_roc_auc'][i] = new_model_results[i][3]
    
    # Calculate the difference between train-test metrics
    model_comp_df['acc_diff'] = abs(model_comp_df['train_acc'] - model_comp_df['test_acc'])
    model_comp_df['roc_auc_diff'] = abs(model_comp_df['train_roc_auc'] - model_comp_df['test_roc_auc'])

    # Reset the index
    model_comp_df.reset_index(inplace=True)

    # Change the original "index" to "classifier"
    model_comp_df.rename(columns={'index':'classifier'}, inplace=True)

    
    return model_comp_df