import os
from tqdm.auto import tqdm

import re
import json

import plotly.express as px
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path


pd.options.plotting.backend = "plotly"


def get_data(path,filetype='csv') :
    """Get the data from its path with a specified format.

    Args:
        path (str): 
        filetype (str, optional): . Defaults to 'csv'.

    Raises:
        ValueError: wrong file type

    Returns:
        DataFrame: 
    """

    if filetype == 'csv':
        return pd.read_csv(path)
    elif filetype == 'tsv':
        return pd.read_csv(path, sep='\t')
    elif filetype == 'json' :
        json_file = json.load(open(path, 'r'))
        return pd.DataFrame(list(json_file.values())[1])
    elif filetype == 'pkl' : 
        return pd.read_pickle(path)
    else:
        raise ValueError("Unsupported file type. Use 'csv', 'tsv', 'json' or 'pkl' .")
    
def df_save_data(df,path,filename,filetype = 'csv',create_folder = True) :
    """Save a dataframe in a specified format.

    Args:
        df (DataFrame): 
        path (str): put the / at the end
        filename (str): 
        filetype (str, optional): csv or pkl. Defaults to 'csv'.

    Raises:
        ValueError: 
    """
    # path = full_path
    full_path = f"{path}{filename}.{filetype}"
    if create_folder  : 
        if os.path.exists(full_path) :
            print('path already exist')
        else :
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if filetype == 'csv' :
            df.to_csv(full_path)
            print(f"Saved CSV to: {full_path}")
        elif filetype == 'json': 
            df.to_json(full_path)
            print(f"Saved json to: {full_path}")


    if filetype == 'csv' : 
        df.to_csv(full_path,index=False)
    elif filetype == 'pkl' :
        df.to_pickle(full_path)

    elif filetype == 'json' :
        df.to_json(full_path)
    else:
        raise ValueError("Unsupported file type. Use 'csv', 'json' or 'pkl'.")

def list_save_data(l,path,filename,filetype='txt') :
    """_summary_

    Args:
        l (list): 
        path (str): put the / at the end
        filename (str): _description_
        filetype (str, optional): Defaults to 'txt'.

    Raises:
        ValueError: wrong filetype
    """
    total_path = f"{path}{filename}.{filetype}"
    if filetype == 'txt' :
        with open(total_path, 'w') as f:
            for item in l:
                f.write(f"{item}\n")
    else : 
        raise ValueError("Unsupported file type. Use 'txt'.")

### DataFrame preprocessing
def set_index(df,col='timestamp',col_name='date',datetime=True,date_format=True,unit='s',timeframe='h',round=True):
    '''
    datetime : if the index is a date index and we want a datetime dtype

    date_format = Bool : If date_format is False the date format is in a timestamp format put unit ='s'
                       Otherwise, it s already with a YYYY-MM-DD ... format
    '''
    df = df.copy()
    if datetime :
        df = to_datetime(df,col,date_format,timeframe,round)
    df = df.set_index(col)
    df.index.rename(col_name,inplace=True)
    return df

def to_datetime(df, col, date_format=True, timeframe='h', round=True, paris_time=False):
    """
    Convert a column or list of columns in a DataFrame to datetime, with optional Paris time localization.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str or list): Column(s) to convert.
        date_format (bool, optional): If True, the format is already in a datetime format. Otherwise, it's in a timestamp format (unit='s'). Defaults to True.
        timeframe (str, optional): Frequency at which to round the time. Defaults to 'h'.
        round (bool, optional): If True, round up the time according to the timeframe. Defaults to True.
        paris_time (bool, optional): If True, localize naive datetimes to Europe/Paris and convert to UTC. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with converted datetime column(s).
    """
    df = df.copy()
    if isinstance(col, list):
        for c in col:
            df = to_datetime(df, c, date_format, timeframe, round, paris_time)
        return df
    else:
        if date_format:
            try:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    df[col] = pd.to_datetime(df[col], format='mixed')
            except:
                def handle_different_format(date):
                    try:
                        res = pd.to_datetime(date, format='mixed')
                    except:
                        try:
                            res = pd.to_datetime(date, format='ISO8601')
                        except:
                            res = pd.NaT
                    return res
                df[col] = df[col].apply(handle_different_format)
            # Localize to Paris and convert to UTC if requested
            if paris_time:
                df[col] = df[col].dt.tz_localize('Europe/Paris', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
            else:
                df[col] = df[col].dt.tz_localize('UTC') if df[col].dt.tz is None else df[col]
            if round:
                df[col] = df[col].dt.ceil(timeframe)
        else:
            df[col] = pd.to_datetime(df[col], unit='s', utc=True)
            if round:
                df[col] = df[col].dt.ceil(timeframe)
        return df


# words_to_remove = ['Protocol Village', 'The protocol', 'First Mover Americas']
# text = 'Protocol Village: Fjord Foundry, a Token-Sale Platform, Raises $4.3M'
# remove_pattern(text, words_to_remove)

def starts_with_sequence(sentence, sequence_list):
    """
    Returns True if the sentence starts with any sequence in sequence_list.
    Args:
        sentence (str): The sentence to check.
        sequence_list (list of str): List of word sequences to match at the start.
    Returns:
        bool: True if sentence starts with any sequence, False otherwise.
    """
    if not isinstance(sentence, str):
        return False
    sentence = sentence.strip().lower()
    for seq in sequence_list:
        if sentence.startswith(seq.lower()):
            return True
    return False






