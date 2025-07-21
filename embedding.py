import importlib 
import utils.processing 
importlib.reload(utils.processing)
from utils.processing import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
# import gensim.downloader as api



# import utils.nlp_utils
# importlib.reload(utils.nlp_utils)
# from utils.nlp_utils import lemmatization_with_spacy,remove_date_pattern, remove_pattern


# from sklearn.metrics import silhouette_score,silhouette_samples
# from joblib import Parallel, delayed

# from tqdm.auto import tqdm
# tqdm.pandas()

# from itertools import chain

# from sklearn.cluster import KMeans
# from sentence_transformers import SentenceTransformer, util
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.cluster import DBSCAN
# import collections
# import hdbscan


# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize


# from langdetect import detect, LangDetectException


# from sklearn.metrics.pairwise import cosine_similarity




### Here we don't want to drop the entire row only to remove more stopping words or stopping sequences
# sequence_to_remove = ['the latest in blockchain tech upgrades, funding announcements and deals. for the period of',
#                       'the protocol','protocol village','first mover americas',
#                       'latest price moves crypto markets context',
#                       'latest blockchain tech upgrades funding announcements deals period', 
#                       'market analysis']

# seq2remove_temporality = ['latest', 'latest price moves', 'daily price moves']

# seq2remove_trend_words = ['bullish', 'bearish', 'bull', 'bear', 'crypto', 'cryptocurrency','tumbles','slumps','drops',
#                             'rallies', 'rising', 'falling','dropping','slumping', 'surges', 'plunges', 'soars', 'plummets','tumbles']

# seq2remove_price = ['price', 'price moves', 'price action', 'price movements','crypto market context']

# seq2remove_vect = [seq2remove_trend_words,
#                    seq2remove_temporality,
#                    seq2remove_price]



 

class VectorizeTxt:
    def __init__(self,df,nb_components=2, pca = True,train=False, df_train=None,max_features=350,**w2v_params):
        self.df_embeddings = df.copy()
        # self.df_embeddings.drop_duplicates(subset=['title','description'], inplace=True)
        self.train = train
        self.df_train = df_train.copy() if df_train != None else df_train
        self.mapping_embedding_method = {'sBert' : self.sentence_bert, 
                                         'TfIdf' : self.tf_idf,
                                         'word2vec' : self.word2vec}
        self.pca = pca
        self.nb_components = nb_components
        self.max_features = max_features
        self.w2v_params = w2v_params

    def sentence_bert(self,col, model_name='all-MiniLM-L6-v2', show_words=False):
        """
        Generate sentence embeddings using Sentence-BERT and optionally print top-n most similar sentences for each row.
        """
        model = SentenceTransformer(model_name)
        sentences = self.df_embeddings[col].fillna('').tolist()
        embeddings = model.encode(sentences, show_progress_bar=True)
        self.embeddings = embeddings
        return embeddings

    def tf_idf(self,col,max_features=350 ,show_words=False):
        tfidf_vect = TfidfVectorizer(max_features=max_features)
        if self.train and self.df_train is not None:
            X_train = self.df_train[col]
            X = self.df_embeddings[col]
            tfidf_vect.fit(X_train)
            embeddings = tfidf_vect.transform(X)
        else:
            X = self.df_embeddings[col]
            embeddings = tfidf_vect.fit_transform(X)

        if show_words:
            n = int(input('Enter the nb of words :'))
            feature_names = np.array(tfidf_vect.get_feature_names_out())
            X_dense = embeddings.toarray()
            X_sort = np.argsort(X_dense, axis=1)[:, -n:]
            relevant_words = np.vectorize(lambda x: feature_names[x])(X_sort)
            print(relevant_words)
        self.embeddings = embeddings.toarray()
        return embeddings
    
    def word2vec(self, col, vector_size=350, window=5, min_count=1, workers=4, show_words=False):
        """
        Generate document embeddings using Word2Vec.
        """
        # Tokenize the sentences
        # col = 'title_desc'
        sentences = self.df_embeddings[col].fillna('').apply(lambda x: simple_preprocess(x)).tolist()
        # Train Word2Vec model
        if self.w2v_params :
            model = Word2Vec(sentences,self.w2v_params)
            vector_size = self.w2v_params['vector_size']
        else : 
            model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        # Generate document embeddings by averaging word vectors
        embeddings = list()
        for sentence in sentences:
            word_vectors = [model.wv[word] for word in sentence if word in model.wv]
            if word_vectors:
                sentence_embedding = np.mean(word_vectors, axis=0).reshape(1,-1)
            else:
                sentence_embedding = np.zeros(vector_size).reshape(1,-1)
            embeddings.append(sentence_embedding)
        # embeddings[0].shape
        # len(embeddings)
        embeddings_arr = np.concatenate(embeddings,axis=0)

        self.embeddings = embeddings_arr
        return embeddings_arr

    def get_pca(self,col,method,show_plot=False):
        # standadization
        scaler = StandardScaler()
        X_std = scaler.fit_transform(self.embeddings)

        if show_plot : 
            # PCA
            pcaa = PCA()
            pcaa.fit(X_std)
            # Explained Variance
            plt.plot(range(1,len(pcaa.explained_variance_ratio_)+1), pcaa.explained_variance_ratio_.cumsum(), marker ='o', linestyle = '--')
            plt.title('Explained Variance by Components')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.show()
        # components
        if self.nb_components == None :
            print(method) 
            nb_components = int(input('How many components do you want? (int):'))
            self.nb_components = nb_components
        else : 
            nb_components = self.nb_components
        
        pcaa = PCA(n_components = nb_components)
        pcaa.fit(X_std)
        scores_pca = pcaa.transform(X_std)
        df_pca = pd.concat([self.df_embeddings, pd.DataFrame(scores_pca,index=self.df_embeddings.index)], axis = 1)
        df_pca.columns = list(self.df_embeddings.columns.values) + [f'feat_emb_{method}_{str(k)}' for k in range(1,nb_components+1)]
        self.df_embeddings = df_pca
    
    def add_embedding(self,col,embedding_method = 'sBERT') :
        embedding_method = [embedding_method] if isinstance(embedding_method,str) else embedding_method
        # col = 'title_desc'
        if isinstance(col,str) :
            # col = c
            # self.clean_to_vectorize(col=col)                                           ##### A retirer
            for method in embedding_method :
                # method = 'word2vec'
                # col= 'title_desc'
                # break
                self.mapping_embedding_method[method](col=col)
                if self.pca  :
                    self.get_pca(col=col,method=method)
                else : 
                    df_embeddings = pd.concat([self.df_embeddings, pd.DataFrame(self.embeddings,index=self.df_embeddings.index)], axis = 1) ### Si on veut les meme index pour df_embeddings et df               
                    df_embeddings.columns = list(self.df_embeddings.columns.values) + [f'feat_emb_{method}_{str(k)}' for k in range(1,self.embeddings.shape[1]+1)]
                    self.df_embeddings = df_embeddings        
        elif isinstance(col,list) : 
            for c in col : 
                # break
                self.add_embedding(col=c,embedding_method = embedding_method)
    
    def get_df(self,col,embedding_method) :
        col = f'clean_emb_{col}'
        self.add_embedding(col,embedding_method)
        mask = self.df_embeddings.columns.str.startswith('emb_')
        self.df_clean = self.df_embeddings.loc[:,~mask]
        return self.df_embeddings

def get_embedding(df_clean_emb,col,emb_models=['word2vec','sBert','TfIdf'],path='./data/embedding/',filename='df_cointelegraph_emb_') : 
    assert f'clean_emb_{col}' in df_clean_emb.columns, 'Need the df_clean_emb dataframe'
    ## Sentiment
    for emb_model in emb_models : 
        print(f'emb {emb_model}')
        vectorizer = VectorizeTxt(df = df_clean_emb,
                                  nb_components = 150,
                                  pca = True,
                                  max_features = 400,)
        df_emb = vectorizer.get_df(col,emb_model)
        df_save_data(df_emb,
                     path,
                     filename+emb_model,
                     filetype = 'csv',
                     create_folder=True)     




if __name__ == '__main__' :

    ## Vectorization
    path = './data/clean_emb/'
    filename = 'df_cointelegraph_clean_emb.csv'

    df_clean_emb = get_data(path = path + filename, filetype = 'csv')
    df_clean_emb = to_datetime(df_clean_emb,col='date',round=False)

    assert ~df_clean_emb.date.duplicated().any(), 'duplicated date'
    assert df_clean_emb.date.is_monotonic_increasing, 'dates are not sorted'
    assert ~df_clean_emb.duplicated(subset=['title_desc']).any(), 'duplicated articles (title_desc col)'


    get_embedding(df_clean_emb,
                  emb_models=['word2vec'],
                  col='title_desc', 
                  path = './data/embedding/', 
                  filename = 'df_cointelegraph_emb_')
    
    