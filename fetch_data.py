# Fetch stock history data from Yahoo Finance

# 1 - select ticker based on year
# 2 - get stock data
# 3 - combine multiple stock data

import pandas as pd
import numpy as np

import yfinance as yf

# 1
def select_ticker(symbols, year):
    '''
    Select ticker only fulfilled the requirement
    year:
    ex. 2011 => that stock has at least 2021-2011 = 10 years of hisory
    '''
    
    timestamps = []
    incep_year = []
    
    for symbol in symbols:
        # Acquire the inception date for each ticker
        timestamps.append(yf.Ticker(symbol).info.get('fundInceptionDate'))
    
    for timestamp in timestamps: 
        # Convert epoch unix to readable dates
        incep_year.append(pd.to_datetime(timestamp, unit='s').year)
    
    # Create a dictionary to pair the ticker with corresponding year
    ticker_dict = dict(zip(symbols, incep_year))
    
    # Create a ticker df
    ticker_df = pd.DataFrame(list(ticker_dict.items()), columns=['symbol', 'start_year'])
    
    # Select desire ticker based on inception date
    new_ticker_df = ticker_df.loc[ticker_df['start_year'] <= year]
    
    new_ticker_list = new_ticker_df['symbol'].tolist()
                                                     
    return new_ticker_list
                                                     
                                                     
                                                     
# 2                                                     
def get_stock_data(symbols):
    '''
    Acquire the historical data from Yahoo Finance
    for provided stock symbol
    '''
    
    # Access ticker data
    ticker = yf.Ticker(symbols)
    
    # Get historical market data
    data = ticker.history(period='max')
    
    return data

# 3
def get_data(symbols):
    '''
    Acquire the data by calling previous function
    Concatenate the dataframe
    '''
    
    # Create a blank dataframe
    df = pd.DataFrame()
    
    for symbol in symbols:
        try:
            # Get all historical market data for all tickers
            df_extra = get_stock_data(symbol)
            
            # Add an extra column to label the ticker
            df_extra['Ticker'] = symbol
            
            # Concatenate all the piece
            df = pd.concat([df,df_extra])
        
        except:
            print(f'Ticker error:{symbol}')
     
    return df