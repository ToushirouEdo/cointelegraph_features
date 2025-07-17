import utils.processing
import importlib
importlib.reload(utils.processing)
from utils.processing import *


from itertools import chain

import re 
from langdetect import detect, LangDetectException


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm




sequence_to_remove = ['the latest in blockchain tech upgrades, funding announcements and deals. for the period of',
                      'the protocol','protocol village','first mover americas',
                      'latest price moves crypto markets context',
                      'latest blockchain tech upgrades funding announcements deals period', 
                      'market analysis']

class TxtCleanerSentiment : 
    def __init__(self) :
        pass

    def cleaning_for_sentiment(self,text):
        text = text.lower()
        text = re.sub(r'@\w+', '@user', text)
        text = re.sub(r'http\S+', 'http', text)
        return ' '.join(text.split())
    
    def clean_column(self,df,col) :
        df = df.copy()
        if isinstance(col,str) :

            df[f'sent_clean_{col}'] = df[col].apply(self.cleaning_for_sentiment)
            return df
        else :
            raise TypeError('col should be a str')
        
class SentimentAnalysis: 
    def __init__(self,sa_model):
        self.sentiment_model = sa_model 
        self.sentiment_models= {'TwitterRoberta' : self.sentiment_Roberta,
                                'FinBert' : self.sentiment_Bert,
                                'CryptoBert' : self.sentiment_Bert,
                                'Vader' : self.sentiment_Vader}
        if sa_model == 'TwitterRoberta' : 
            ## Model
            self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            # self.model_name = 'pysentimiento/robertuito-sentiment-analysis'
        if sa_model == 'FinBert' :
            # self.model_name = 'yiyanghkust/finbert-tone'ß
            self.model_name = 'ProsusAI/finbert' ## Better
        if sa_model == 'CryptoBert':
            self.model_name = 'ElKulako/cryptobert'
        if sa_model == 'GPT2' :   ## pas acces
            self.model_name = 'mrm8488/gpt2-finetuned-sentiment'

        try :
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except :
            pass     

    def get_score_3labels(self,df) : 
        eps = 1e-12
        df.loc[:,'feat_sent_score'] = df['feat_sent_positive'] - df['feat_sent_negative']
        df.loc[:,'feat_sent_score_biased_center'] = (df['feat_sent_positive'] - df['feat_sent_negative']) / (1 - df['feat_sent_neutral'] + eps)
        df.loc[:,'feat_sent_score_sigmoid_adj']  = (df['feat_sent_positive'] - df['feat_sent_negative']) / (df['feat_sent_positive'] + df['feat_sent_negative'] + eps)
        df.loc[:,'feat_sent_entropy'] = (- df[['feat_sent_negative','feat_sent_neutral','feat_sent_positive']] * np.log(df[['feat_sent_negative','feat_sent_neutral','feat_sent_positive']]+eps)).sum(1)
        df.loc[:,'feat_sent_score_entropy_adj'] = (df['feat_sent_positive'] - df['feat_sent_negative']) * (1- df['feat_sent_entropy'])
        return df

    def sentiment_Roberta(self,df,col):
        # df=df_clean_sa.iloc[:10]
        # df = df_clean_sa
        df = df.copy()
        labels = ['feat_sent_negative', 'feat_sent_neutral', 'feat_sent_positive']
        def get_sentiment_text(text,labels) :
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,max_length=512)
            with torch.no_grad():
                output = self.model(**inputs)
                output = output[0][0].detach().numpy()
                probs = softmax(output)
            return pd.Series(probs,index=labels)
        print(col)
        print(df.columns)
        df[labels] = df[col].progress_apply(lambda text : get_sentiment_text(text,labels))
        df = self.get_score_3labels(df)
        return df    
    ## FinBert bad
    def sentiment_Bert(self, df, col):
        df = df.copy()
        labels = ['feat_sent_positive', 'feat_sent_neutral', 'feat_sent_negative']  # FinBERT label order

        def get_sentiment_text(text,labels):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0].detach().numpy()
                probs = softmax(logits)
            return pd.Series(probs, index=labels)
        df[labels] = df[col].progress_apply(lambda text : get_sentiment_text(text,labels))
        df = self.get_score_3labels(df)  
        return df
    
    def sentiment_Vader(self, df, col):
        self.model = SentimentIntensityAnalyzer()
        df = df.copy()
        labels = ['feat_sent_negative', 'feat_sent_neutral', 'feat_sent_positive','feat_sent_compound']
        def get_sentiment_text(text,labels):
            # text = 'It was amazing yesterday'
            scores = self.model.polarity_scores(text)
            return pd.Series(
                [scores['neg'], scores['neu'], scores['pos'],scores['compound']],
                index=labels
            )
        df[labels] = df[col].progress_apply(lambda text : get_sentiment_text(text,labels))
        df = self.get_score_3labels(df)  
        return df

    def get_sentiment(self,df,col) :
        col = f'sent_clean_{col}'
        df = self.sentiment_models[self.sentiment_model](df,col)
        return df
  
def get_sentiment(df_text_clean,col,sa_models=['TwitterRoberta','CryptoBert','FinBert','Vader'],path='./data/sentiment/',filename='df_cointelegraph_sent_') : 
    from sentiment_analysis import sequence_to_remove ,TxtCleanerSentiment, SentimentAnalysis
    ## Txt cleaning
    print("Txt cleaner sentiment analysis")
    txt_cleaner_sa = TxtCleanerSentiment()
    df_clean_sa = txt_cleaner_sa.clean_column(df_text_clean,col)
    
    ## Sentiment
    for sa_model in sa_models : 
        print(f'sentiment features {sa_model}')
        sentiment_analyzer = SentimentAnalysis(sa_model='CryptoBert')
        df_sa_features =  sentiment_analyzer.get_sentiment(df_clean_sa,col)
        df_save_data(df_sa_features,
                     path,
                     filename+sa_model,
                     filetype = 'csv',
                     create_folder=True)     


if __name__ == '__main__' : 

    path = './data/df_cointelegraph_text_clean.csv'
    df_text_clean = get_data(path,filetype='csv')
    df_text_clean = to_datetime(df_text_clean,col='date')
    # df_test = df_text_clean.iloc[:10]
    get_sentiment(df_text_clean,col='title_desc')


    
    


    


    