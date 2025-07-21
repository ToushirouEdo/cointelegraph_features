import importlib 
import utils.processing 
importlib.reload(utils.processing)
from utils.processing import get_data, df_save_data

import utils.nlp_utils
importlib.reload(utils.nlp_utils)
from utils.nlp_utils import remove_pattern


import pandas as pd 
from datetime import datetime
from bs4 import BeautifulSoup
import re 
from langdetect import detect, LangDetectException


def preprocessing_cointelegraph(df,cols_kept,col_date='date',paris_time=False) :
    df = df[cols_kept].copy()  
    ## String columns
    df[['title','subtitle','url','author']] = df[['title','subtitle','url','author']].astype(dtype='string')
    ## Author
    df['author'] = df['author'].apply(lambda author : remove_pattern(author,words_to_remove=['Cointelegraph by']) if not pd.isna(author) else pd.NA) 
    ## Add a source column 
    df['source'] = 'Cointelegraph'    
    ##Date
    def cointelegraph_datetime(date_str) : 
        try :
            return datetime.strptime(date_str,"%a, %d %b %Y %H:%M:%S %z")
        except: 
            return datetime.strptime(date_str,"%Y-%m-%d %H:%M:%S")
                
    df[col_date] = df[col_date].apply(cointelegraph_datetime)
    df[col_date] = pd.to_datetime(df[col_date],utc=True) ## +1 =
    # df.date.iloc[1379:1381]
    # pd.to_datetime(df[col_date].iloc[1378:1380])
    # pd.to_datetime(df[col_date].iloc[1381])
    # pd.to_datetime(df['date'],format='mixed')
    ## Description    
    df['subtitle']= df['subtitle'].apply(lambda html : BeautifulSoup(html, "html.parser").get_text(strip=True))
    df['title_desc'] = df['title'] + df['subtitle']
    return df 


sequence_to_remove = ['the latest in blockchain tech upgrades, funding announcements and deals. for the period of',
                      'the protocol','protocol village','first mover americas',
                      'latest price moves crypto markets context',
                      'latest blockchain tech upgrades funding announcements deals period', 
                      'market analysis']


### We can add an argue to remove the rows that starts with a certain sequence or countain a certain sequence
class TextCleaner:
    """Renvoie df sans les lignes d'une autre langue que l'anglais, et trier selon les dates de publications
    """
    def __init__(self,sequence_to_remove=sequence_to_remove):
        self.sequence_to_remove = sequence_to_remove

    def keep_english_rows(self, df, col = 'title_desc'):
        def is_english(text):
            try:
                return detect(text) == 'en'
            except LangDetectException:
                return False
        mask = df[col].fillna('').apply(is_english)
        return df[mask].copy()

    def clean_text(self, text):
        if pd.isna(text):
            return ''
        # Lowercase
        
        try :
            text = text.lower()
        except : 
            pass
        text = remove_pattern(text, self.sequence_to_remove)
        ## Remove things 
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove emails
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', ' ', text)
        # Remove special signs (€, $, £, ¥, etc.)
        text = re.sub(r'[\$€£¥₹¢₩₽₺₴₦₱₲₵₡₢₫₭₮₯₠₣₤₥₧₨₩₪₫₭₮₯₰₱₲₳₴₵₸₺₼₽₾₿]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()

        return ' '.join(tokens)

    def clean_column(self,df, col):
        df = self.keep_english_rows(df).copy()
        if isinstance(col,str) :
            df[col] = df[col].apply(self.clean_text)
        elif isinstance(col,list) : 
            df[col] = df[col].map(self.clean_text)
        else : 
            raise TypeError('col should be a str or a list')
        df = df.drop_duplicates(subset='title_desc')
        df.sort_values(by='date',inplace=True)
        df = df.drop_duplicates(subset='date')
        df.loc[:,'idx_news'] = df.index
        return df
    


if __name__ == '__main__' : 
    ## Dylan file 
    path = './data/dataset/'
    filename = 'cointelegraph_rss_wayback.csv'
    df_cointelegraph = get_data(path+filename,filetype='csv')

    ## Preprocessing
    cols_kept = ['date','url','author','title','subtitle']
    df_cointelegraph = preprocessing_cointelegraph(df_cointelegraph,cols_kept)

    txt_cleaner = TextCleaner()
    df_clean = txt_cleaner.clean_column(df_cointelegraph,col='title_desc')
    assert df_clean.date.is_monotonic_increasing, 'no sorted by date'

    df_save_data(df_clean,
                 path='./data/text_clean/',
                 filename = 'df_cointelegraph_text_clean',
                 filetype = 'csv')
    
