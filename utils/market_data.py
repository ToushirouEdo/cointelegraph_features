import pandas as pd
import time
from datetime import datetime, timedelta
import requests

import argparse

import importlib
import processing 
importlib.reload(processing)
from processing import *


def get_btc_history(symbol='BTCUSDT', interval='1m', start_time='2022-06-01', end_time=None, limit=1000):
    """
    Fetch historical minute-level BTC price data from Binance.

    Args:
        symbol (str): Trading pair (default 'BTCUSDT').
        interval (str): Candlestick interval (only '1m' supported here).
        start_time (str): Start date as 'YYYY-MM-DD'.
        end_time (str): End date as 'YYYY-MM-DD'. Defaults to now if None.
        limit (int): Max results per request (max 1000).

    Returns:
        pd.DataFrame: Historical BTC minute-level price data.
    """
    url = "https://api.binance.com/api/v3/klines"
    all_data = []

    # Convert string dates to datetime
    start_time = datetime.strptime(start_time, "%Y-%m-%d")
    end_time = datetime.strptime(end_time, "%Y-%m-%d") if end_time else datetime.utcnow()

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    while start_ms < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': limit
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data.extend(data)

        # Move to next batch (advance by 1 minute past the last time)
        start_ms = data[-1][0] + 60_000
        time.sleep(0.3)  # Respect rate limits

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])

    # Then add 'date' column
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")

    # Now subset the columns you want:
    df = df[["date", "open", "high", "low", "close", "volume"]]

    # Convert to float where needed
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df

def add_forward_return(df, df_ohlc, n_minutes=5,col_name = None):
    """
    Adds a forward return column to df based on open prices from df_ohlc.
    
    Parameters:
        df (pd.DataFrame): DataFrame with a datetime column for news timestamps.
        df_ohlc (pd.DataFrame): Minutely OHLC data with datetime index or column.
        n_minutes (int): Number of minutes ahead to calculate return from t+1 minute.
        'date' (str): Name of the datetime column in df.
        'date' (str): Name of the datetime column in df_ohlc (if not index).

    Returns:
        pd.DataFrame: Original df with an added 'future_return_nmin' column.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    
    # Ensure OHLC datetime is parsed and set as index
    ohlc = df_ohlc.copy()
    ohlc['date'] = pd.to_datetime(ohlc['date'])
    ohlc.set_index('date', inplace=True)
    ohlc.sort_index(inplace=True)

    # Function to compute return per row
    def get_return(row,n_minute):
        # Round to the next minute (ceil)
        t0 = row['date'].ceil(freq='min')
        t1 = t0 + pd.Timedelta(minutes=n_minute)

        try:
            open_start = ohlc.loc[t0]['open']
            open_end = ohlc.loc[t1]['open']
            return (open_end / open_start) - 1
        except KeyError:
            return pd.NA
    assert isinstance(n_minutes,int) or isinstance(n_minutes,list) , 'n_minutes has to be a list or a int'
    if isinstance(n_minutes,int) :
        n_minutes = [n_minutes]
    for n_minute in n_minutes : 
        if col_name == None :
            df[f'feat_ret_{n_minute}min'] = df.apply(lambda row : get_return(row,n_minute), axis=1)
        else :
            df[f'{col_name}_{n_minute}min'] = df.apply(lambda row : get_return(row,n_minute), axis=1)
    return df

def main(start, end):
    btc = get_btc_history(start_time=start, end_time=end, interval='1m')
    path = './data/market_data/'
    filename = f'df_btc_1m_{start}_{end}'
    df_save_data(btc, path, filename, 'json', create_folder=True)


if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()
    main(args.start, args.end)