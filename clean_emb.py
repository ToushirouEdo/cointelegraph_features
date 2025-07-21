import importlib


import utils.processing
importlib.reload(utils.processing)
from utils.processing import *

import utils.nlp_utils
importlib.reload(utils.nlp_utils)
from utils.nlp_utils import lemmatization_with_spacy,remove_date_pattern, remove_pattern

from itertools import chain


### Here we don't want to drop the entire row only to remove more stopping words or stopping sequences
sequence_to_remove = ['the latest in blockchain tech upgrades, funding announcements and deals. for the period of',
                      'the protocol','protocol village','first mover americas',
                      'latest price moves crypto markets context',
                      'latest blockchain tech upgrades funding announcements deals period', 
                      'market analysis']

seq2remove_temporality = ['latest', 'latest price moves', 'daily price moves']

seq2remove_trend_words = ['bullish', 'bearish', 'bull', 'bear', 'crypto', 'cryptocurrency','tumbles','slumps','drops',
                            'rallies', 'rising', 'falling','dropping','slumping', 'surges', 'plunges', 'soars', 'plummets','tumbles']

seq2remove_price = ['price', 'price moves', 'price action', 'price movements','crypto market context']

seq2remove_vect = [seq2remove_trend_words,
                   seq2remove_temporality,
                   seq2remove_price]



class TextCleanerClustering : 
    def __init__(self,df,col,seq2remove_vect,lemmatization=True,remove_date=True):
        self.df = df.copy()
        self.df[col] = df[col].astype(dtype='string')
        self.col = col
        self.seq2remove_vect = seq2remove_vect
        if any(isinstance(i, list) for i in self.seq2remove_vect):
            self.seq2remove_vect = list(chain.from_iterable(self.seq2remove_vect))
        self.lemmatization = lemmatization
        self.remove_date = remove_date
    
    def clean_text(self, text):
        if pd.isna(text):
            return ''
        # Lowercase
        
        try :
            text = text.lower()
        except : 
            pass
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

    def clean_to_vectorize(self, n_jobs=-1):
        series =  self.df[self.col].apply(self.clean_text)
        if self.lemmatization:
            print('lemmatization embeddings')
            # tqdm for progress bar
            series = self.df[self.col].apply(lemmatization_with_spacy)

        if self.seq2remove_vect:
            print('Removing sequences')
            series = series.apply(lambda txt : remove_pattern(txt,self.seq2remove_vect))
            # series = Parallel(n_jobs=n_jobs)(
            #     delayed(remove_pattern)(text, self.seq2remove_vect) for text in tqdm(series, desc="Remove pattern")
            # )
        if self.remove_date:
            print('removing date')
            series = series.apply(lambda txt : remove_date_pattern(txt))
            # series = Parallel(n_jobs=n_jobs)(
            #     delayed(remove_date_pattern)(text) for text in tqdm(series, desc="Remove date")
            # )
        self.df[f'clean_emb_{self.col}'] = series
        return self.df
    

def get_clean_clust(df,col,path,filename) :
    assert 'date' in df.columns , 'date not in df_clust'
    assert df.date.is_monotonic_increasing, 'df_clust is not sorted by date'
    assert not (df.date.duplicated().any()), 'duplicated date'
    assert df.date.is_monotonic_increasing, 'dates are not sorted'
    assert not (df.duplicated(subset=['title_desc']).any()), 'duplicated articles (title_desc col)'
    cleaner_clustering = TextCleanerClustering(df,
                                               col,
                                               seq2remove_vect=seq2remove_vect,
                                               lemmatization=True,
                                               remove_date=True)

    df_clean_clust  = cleaner_clustering.clean_to_vectorize()
    df_save_data(df_clean_clust,
                 path,
                 filename,
                 filetype='csv',
                 create_folder=True)





if __name__ == '__main__' : 

    path = './data/text_clean/df_cointelegraph_text_clean.csv'
    df_text_clean = get_data(path,filetype='csv')
    df_text_clean = to_datetime(df_text_clean,col='date')
    # df_test = df_text_clean.iloc[:2000]

    path_clean_clust = './data/clean_emb/'
    filename_clean_clust='df_cointelegraph_clean_emb'
    # get_clean_clust(df_text_clean,col='title_desc')
    get_clean_clust(df_text_clean,
                    col='title_desc',
                    path = path_clean_clust, 
                    filename = filename_clean_clust)
