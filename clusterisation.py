import importlib 

import utils.processing 
importlib.reload(utils.processing)
from utils.processing import *

import utils.market_data 
importlib.reload(utils.market_data )
from utils.market_data  import add_forward_return




from sklearn.metrics import silhouette_score,silhouette_samples,pairwise_distances 
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans, DBSCAN, OPTICS


def test_rigth_dataframe(df_sent,df_emb) : 
    assert 'date' in df_emb.columns , 'date not in df_clust'
    assert df_emb.date.is_monotonic_increasing, 'df_clust is not sorted by date'
    assert not (df_emb.date.duplicated().any()), 'duplicated date'
    assert df_emb.date.is_monotonic_increasing, 'dates are not sorted'
    assert not (df_emb.duplicated(subset=['title_desc']).any()), 'duplicated articles (title_desc col)'

    assert 'date' in df_sent.columns , 'date not in df_clust'
    assert df_sent.date.is_monotonic_increasing, 'df_clust is not sorted by date'
    assert not (df_sent.date.duplicated().any()), 'duplicated date'
    assert df_sent.date.is_monotonic_increasing, 'dates are not sorted'
    assert not(df_sent.duplicated(subset=['title_desc']).any()), 'duplicated articles (title_desc col)'

    assert len(df_sent) == len(df_emb) , 'different len df_sent and df_emb'
    assert (df_sent['title_desc'].values == df_emb['title_desc'].values).all()
    assert (df_sent['date'].values == df_emb['date'].values).all()

class GetClusteringParamsKMeans:
    def __init__(self,X,metric, method, **kwargs):
        self.X = X
        self.metric = metric
        self.method = method
        self.kwargs = kwargs
        self.opti_method = {
            'naive': self.naive_sqrt_n,
            'silhouette_score': self.opti_silhouette_score_kmeans
        }

    def naive_sqrt_n(self):
        """
        Naive heuristic: number of clusters = sqrt(n)
        """
        n_samples = len(self.X)
        k = max(2, int(np.sqrt(n_samples)))
        return {'k': k}

    def opti_silhouette_score_kmeans(self, range_k):
        """
        Optimize k by maximizing the silhouette score.
        
        Args:
            range_k: list or range of candidate cluster numbers (e.g., range(2, 15))
        """
        best_k = None
        best_score = -10

        for k in range_k:
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=0).fit(self.X)
            labels = kmeans.labels_
            score = silhouette_score(self.X, labels,metric = self.metric)

            if score > best_score:
                best_score = score
                best_k = k

        if best_k is None:
            print("No optimal k found, returning default")
            return {'n_clusters': 2}

        return {'k': best_k}

    def get_params(self):
        return self.opti_method[self.method](**self.kwargs)

class GetClusteringParamsDBSCAN : 
    def __init__ (self,X,metric,method,**kwargs) : 
        self.X = X
        self.metric = metric 
        self.method = method
        self.kwargs = kwargs
        self.opti_method = {'nb_cluster' : self.opti_min_eps_max_nb_cluster, 
                            'silhouette_score' : self.opti_silhouette_score_dbscan}

    def opti_min_eps_max_nb_cluster(self,range_eps,size_min) :
        res_eps = None
        nb_labels_max = -10
        for eps in range_eps: 
            clustering = DBSCAN(eps=eps, min_samples=size_min, metric=self.metric).fit(self.X)
            labels = clustering.labels_
            nb_labels = len(set(labels))
            if nb_labels > nb_labels_max : 
                res_eps = eps 
                nb_labels_max = nb_labels
        if (res_eps is None) : 
            print("pas d'opti")
            return dict(eps=0.1,
                        size_min=5)
        return {'eps' :res_eps, 'size_min' : size_min}
        
    def opti_loop_over_eps_dbscan2(self,size_min=5) :
        """ dbscan parameters optimizer : 
            fix min_size
            find the smallest eps that given the larger number of clusters

        Args:
            eps_interval (tuple ): min_eps, max_eps
        """
        eps = 0.1
        clustering = DBSCAN(eps=eps, min_samples=size_min, metric=metric).fit(self.X)
        labels = clustering.labels_
        unique_label = set(np.unique(labels))

        while not ((len(unique_label) == 1) & (-1 in unique_label)) : 
            eps = eps / 10
            clustering = DBSCAN(eps=eps, min_samples=size_min, metric=metric).fit(self.X)
            labels = clustering.labels_
            unique_label = set(np.unique(labels))
        
        res_eps = eps
        eps_min = eps
        previous_nb_label = len(unique_label)
        for eps in np.arange(eps_min,1,eps_min*0.1) :
            clustering = DBSCAN(eps=eps, min_samples=size_min, metric=metric).fit(self.X)
            labels = clustering.labels_
            nb_label = len(set(np.unique(labels)))
            if nb_label > previous_nb_label : 
                res_eps = eps
                previous_nb_label = nb_label
        return {'eps':res_eps, 'size_min':size_min}
                         
    def opti_silhouette_score_dbscan(self,range_eps,range_size_min) :
        res_eps = None
        res_size_min = None
        max_silhouette = -10
        for eps in range_eps : 
            for size_min in range_size_min  :
                # eps = 0.2
                # size_min = 5
                clustering = DBSCAN(eps=eps, min_samples=size_min, metric=self.metric).fit(self.X)
                labels = clustering.labels_
                X = self.X[labels != -1].copy()
                labels = labels[labels != -1].copy()
                if len(set(labels)) > 1 :
                    current_silhouette = silhouette_score(X,labels,metric=self.metric)
                    if max_silhouette < current_silhouette : 
                        max_silhouette = current_silhouette
                        res_eps = eps 
                        res_size_min = size_min
        if (res_eps is None) & (res_size_min is None) : 
            print("pas d'opti")
            return dict(eps=0.1,
                        size_min=5)
        return {'eps' :res_eps, 'size_min' :res_size_min}
   
    def get_params(self) :
        # method = 'opti_loop_over_eps_dbscan'
        params = self.opti_method[self.method](**self.kwargs)
        return params

class GetCluster : 
    def __init__(self,X,method,metric):
        self.X = X
        self.method = method
        self.metric = metric
        self.clustering_algos = {'kmeans' : self.kmeans,
                                 'dbscan' : self.dbscan,
                                 'optics' : self.optics}
        
    def dbscan(self,eps,size_min) :
        clustering = DBSCAN(eps=eps, min_samples=size_min, metric=self.metric).fit(self.X)
        self.labels = clustering.labels_
   
    def kmeans(self,k) : 
        clustering = KMeans(n_clusters=k, n_init=20, random_state=0).fit(self.X)
        self.labels = clustering.labels_
    
    def optics(self,size_min): 
        clustering = OPTICS(min_samples=size_min,metric=self.metric).fit(self.X)
        self.labels = clustering.labels_

    def get_cluster(self,**params) : 
        self.clustering_algos[self.method](**params)
        self.unique_labels = np.unique(self.labels)
        self.nb_labels = len(set(self.unique_labels))
        return self.labels , self.unique_labels,self.nb_labels

class AddFeaturesClusterWindow2: 

    def __init__(self,
                 df_window, 
                 X,
                 date,
                 clust_algo='dbscan',
                 metric='cosine',
                 opti=True,
                 opti_method=None,
                 opti_params = None,
                 clustering_params = None):
        self.df_window = df_window.copy() 
        self.X = X
        self.date = date
        self.clust_algo = clust_algo
        self.metric = metric
        ## Optimization  
        self.opti = opti
        self.opti_method = opti_method
        self.opti_params = opti_params
        self.opti_algos = {'kmeans':GetClusteringParamsKMeans,
                           'dbscan' : GetClusteringParamsDBSCAN}
        ## Hard coded params
        self.clustering_params = clustering_params
        ## 

    def get_cluster(self) : 
        ## Optimization
        if self.opti :
            opti_algo = self.opti_algos[self.clust_algo](X = self.X,
                                                                metric = self.metric,
                                                                method = self.opti_method,
                                                                **self.opti_params)
            
            # self = self.opti_algos[self.clust_algo](X = self.X,
            #                                                     metric = self.metric,
            #                                                     method = self.opti_method,
            #                                                     **self.opti_params)
            params = opti_algo.get_params()   
            
            
            ## dict avec le nom des params
            self.clustering_params = params
        
        ## Clusterization
        clustering_algo = GetCluster(X=self.X,
                                        method =self.clust_algo,
                                        metric = self.metric)
        clustering_algo.get_cluster(**self.clustering_params)
        self.labels = clustering_algo.labels
        self.unique_labels = clustering_algo.unique_labels
        self.nb_labels = clustering_algo.nb_labels      
  
    def get_nb_cluster(self) :
        nb_cluster  = [np.nan] * self.nb_labels + [self.nb_labels]
        nb_cluster_df = pd.DataFrame(nb_cluster,columns=['nb_cluster'],index = self.idx)
        self.res = pd.concat([self.res,nb_cluster_df],axis=1)
    
    def get_nb_news(self) : 
        labels, counts = np.unique(self.labels,return_counts=True)
        nb_news_per_cluster = pd.Series(counts,index =labels)
        nb_news_per_cluster.loc['tot'] = np.sum(counts)

        idx = pd.MultiIndex.from_product(
                [[self.date], nb_news_per_cluster.index],
                names=['date', 'label']
        )
        nb_df = pd.DataFrame(
            nb_news_per_cluster.values,
            index=idx,
            columns=['nb_news']
        )
        self.res = pd.concat([self.res,nb_df],axis=1)
    
    def get_silhouette_score(self) :
        ## Remove outlier
        X = self.X[self.labels != -1]
        labels = self.labels[self.labels != -1]

        ## Si on a un nb de cluster == 1 on met des nan
        if len(set(labels)) == 1 :
            stat_intra_df = pd.DataFrame([[np.nan]] *len(self.res) ,index = self.res.index ,columns= ['silhouette_score'])
            self.res = pd.concat([self.res ,stat_intra_df],axis=1)
            return
        
        silhouette_per_cluster = dict()
        silhouette_score_array = silhouette_samples(X,labels,metric=self.metric)
        for i in list(set(labels)):
            silhouette_per_cluster[i] = np.mean(silhouette_score_array[labels == i])
        silhouette_per_cluster[-1] = np.nan    ## attention a cette ligne 
        silhouette_per_cluster['tot'] = np.mean(silhouette_score_array)

        silhouette_per_cluster = pd.Series(silhouette_per_cluster)

        idx = pd.MultiIndex.from_product(
                [[self.date], silhouette_per_cluster.index],
                names=['date', 'label']
        )

        silhouette_df = pd.DataFrame(
            silhouette_per_cluster.values,
            index=idx,
            columns=['silhouette_score']
        )
        self.res = pd.concat([self.res,silhouette_df],axis=1)

    def get_centroid(self) : 
        centroid_dict = dict()
        for i in self.unique_labels : 
            centroid_dict[i] = np.mean(self.X[self.labels == i],axis=0)
        centroid_dict['tot'] = np.mean(self.X,axis=0)
        centroid_df = pd.DataFrame(centroid_dict,index = [f'emb_{k}' for k in range(1,self.X.shape[1]+1)])
        centroid_df = centroid_df.T
        idx = pd.MultiIndex.from_product(
            [[self.date], centroid_df.index],
            names=['date', 'label']
        )
        centroid_df.index = idx
        self.res = pd.concat([self.res,centroid_df],axis=1)
        self.centroid_df = centroid_df
    
    def get_mean_std_distance_to_centroid(self) :
        mean_dist_cluster_dict = dict()
        std_dist_cluster_dict = dict()
        for i in self.unique_labels :
            centroid = self.centroid_df.xs(i,level='label')
            X_cluster = self.X[self.labels == i]
            mean_dist_cluster_dict[i] = np.mean(pairwise_distances(X_cluster,centroid,metric=self.metric))
            std_dist_cluster_dict[i] = np.std(pairwise_distances(X_cluster,centroid,metric=self.metric))
        
        mean_dist_per_cluster = pd.Series(mean_dist_cluster_dict)
        std_dist_per_cluster = pd.Series(std_dist_cluster_dict)
        idx = pd.MultiIndex.from_product(
                [[self.date], mean_dist_per_cluster.index],
                names=['date', 'label']
        )
        mean_dist_df = pd.DataFrame(
            mean_dist_per_cluster.values,
            index=idx,
            columns=['mean_dist']
        )
        std_dist_df = pd.DataFrame(
            std_dist_per_cluster.values,
            index=idx,
            columns=['std_dist']
        )
        self.res = pd.concat([self.res,mean_dist_df,std_dist_df],axis=1)
    
    def get_sent_feat(self) :
        sent_mask = [c for c in self.df_window.columns  if c.startswith('feat_sent')]
        
        sent_df_mean = self.df_window.groupby(by='label')[sent_mask].mean()
        sent_df_mean.loc['tot',:] = self.df_window[sent_mask].mean()
        sent_df_mean.columns = [c + '_mean' for c in sent_df_mean.columns]

        sent_df_std = self.df_window.groupby(by='label')[sent_mask].std()
        sent_df_std.loc['tot',:] = self.df_window[sent_mask].std()
        sent_df_std.columns = [c + '_std' for c in sent_df_std.columns]
        
        sent_df = pd.concat([sent_df_mean,sent_df_std],axis=1)
        sent_df.index = self.idx
        self.res = pd.concat([self.res,sent_df],axis=1)

    def get_market_data_feat(self) : 
        ret_mask = [c for c in self.df_window.columns  if c.startswith('feat_ret')]
        
        ret_df_mean = self.df_window.groupby(by='label')[ret_mask].mean()
        ret_df_mean.loc['tot',:] = self.df_window[ret_mask].mean()
        ret_df_mean.columns = [c + '_mean' for c in ret_df_mean.columns]

        if self.df_window[ret_mask].isna().any().any() :
            ret_df_std = pd.DataFrame(np.nan , index = self.unique_labels,columns=ret_mask) 
        else :
            ret_df_std = self.df_window.groupby(by='label')[ret_mask].std()
            ret_df_std.loc['tot',:] = self.df_window[ret_mask].std()
        ret_df_std.columns = [c + '_std' for c in ret_df_std.columns]
        
        ret_df = pd.concat([ret_df_mean,ret_df_std],axis=1)
        ret_df.index = self.idx
        self.res = pd.concat([self.res,ret_df],axis=1)

    def get_features(self,refit=True) : 
        if refit : 
            self.get_cluster()
            
        self.df_window.loc[:,'label'] = self.labels 
        self.idx  = pd.MultiIndex.from_product([[self.date],list(self.unique_labels) + ['tot']],names=['date','label'])
        self.res = pd.DataFrame(index=self.idx)
        self.get_nb_cluster()
        self.get_nb_news()
        self.get_sent_feat()
        self.res
        self.get_market_data_feat()
        self.get_silhouette_score()
        ## add params as features
        if 'eps' in self.clustering_params.keys() :
            self.res.loc[slice(None),'eps'] = [self.clustering_params.get('eps',np.nan)] * len(self.res)
        if 'size_min' in self.clustering_params.keys() : 
            self.res.loc[slice(None),'size_min'] = [self.clustering_params.get('size_min',np.nan)] * len(self.res)
        if 'k' in self.clustering_params.keys() : 
            self.res.loc[slice(None),'k'] = [self.clustering_params.get('k',np.nan)] * len(self.res)

        self.get_centroid()
        self.get_mean_std_distance_to_centroid()
        news_label = self.labels[-1]

        ## Mettre les indicateurs ici que l'on veut
        cluster_feat_col =  ['nb_news','silhouette_score','mean_dist','std_dist'] + [c for c in self.res.columns if c.startswith('emb_')]
        if 'size_min' in self.res.columns : 
            cluster_feat_col = ['size_min'] + cluster_feat_col

        if 'eps' in self.res.columns :
            cluster_feat_col =  ['eps' ] + cluster_feat_col

        general_feat_col = ['nb_cluster','nb_news','silhouette_score']
           
        ## sentiment feat
        sent_feat_col = [c for c in self.res.columns  if c.startswith('feat_sent')]

        ## ret feat
        ret_feat_col = [c for c in self.res.columns  if c.startswith('feat_ret')]        

        if any(col not in self.res.columns for col in ['eps', 'size_min']):
            print('eps or size_min not in clusering params')
        
        cluster_feats = self.res.xs(news_label,level='label')[cluster_feat_col + sent_feat_col + ret_feat_col]
        general_feats = self.res.xs('tot',level='label')[general_feat_col + sent_feat_col + ret_feat_col]
        general_feats.columns =[f'{c}_tot' for c in general_feats.columns]

        self.res = pd.concat([cluster_feats,general_feats],axis=1)
        self.res.loc[:,'label'] = news_label
        self.res
    
## At each news we refit the clusters
class AddFeaturesCluster2 : 
    """opti_method : 'opti_silhouette_score_dbscan'
    """
    def __init__(self,
                 df_embedding, ## doit avoir des datetimes
                 emb_model, 
                 clust_algo,
                 metric = 'cosine',
                 opti = True, 
                 opti_method = None,
                 opti_params = None,
                 clustering_params = None,
                 lookback_window = 7,   ## nb of days
                 min_nb_news = 10,
                 max_forward_looking = 0
                 ) :    
        self.df_embedding = df_embedding
        if self.df_embedding['date'].duplicated().any() :
            print('Duplicated date in df_embedding. Drop duplicated automatically')
            self.df_embedding.drop_duplicates(subset='date',inplace=True)
        if not (self.df_embedding.date.is_monotonic_increasing) : 
            print('df is not sorted by date. Sort it automatically')
            self.df_embedding.sort_values(by='date')

        
        self.emb_model = emb_model          ## TfIdf, sBert, word2vec 
        ## Clustering
        self.clust_algo = clust_algo
        self.metric = metric 
        #### Clustering algo params
        ## Opti
        self.opti = opti 
        self.opti_method = opti_method 
        self.opti_params = opti_params
        ## Hard coded params
        self.clustering_params = clustering_params

        #### Lookback 
        self.lookback_window = lookback_window
        # Nb news minimum over the timeframe to take account that date
        self.min_nb_news = min_nb_news

        # Nb of minute we're looking forward to get the return after the news
        self.max_forward_looking = pd.Timedelta(minutes = max_forward_looking)

        ## DataFrame with the clustering features
        self.df_res = pd.DataFrame()

        ## Prev model
        self.prev_date = None 
        self.prev_labels = None 
        self.prev_df_window = None
        self.prev_model = None

    def get_X(self,df) : 
        mask = [c for c in df.columns if c.startswith(f'feat_emb_{self.emb_model}')]
        X = df[mask].values
        return X

    def clusterisation_without_refit(self,df_window,X_window,date):

        prev_X = X_window[:-1]
        current_X = X_window[-1:]
        prev_labels = self.prev_model.labels[-(len(prev_X)):]
  
        # X_window.shape
        # self.prev_model.labels.shape
        # prev_X.shape
        # current_X.shape
        # prev_labels.shape

        # X_window.shape

        eps = self.prev_model.clustering_params.get('eps')
        if eps == None  : 
            raise('eps is not in prev_model clustering_params')
        size_min = self.prev_model.clustering_params.get('size_min')

        self.clustering_params = self.prev_model.clustering_params

        mask_eps = np.where(pairwise_distances(prev_X,current_X) < eps)[0]
        if len(mask_eps) == 0 :   ## si on met un mask vide on a recupere tous les labels different de 1  
            current_label = -1
        else :
            neighbors = prev_labels[prev_labels!=-1][mask_eps]
            if neighbors.size == 0 : 
                current_label = -1
            else : 
                labels, counts = np.unique(neighbors,return_counts=True)
                if counts > size_min : 
                    current_label = labels[np.argmax(counts)]
                else :
                    current_label = -1


        labels = np.concatenate([prev_labels,np.array([current_label])],axis=0)

        current_model = AddFeaturesClusterWindow2(df_window=df_window,
                                                  X = X_window,
                                                date = date,
                                                clust_algo = self.clust_algo,
                                                metric = self.metric,
                                                opti = self.opti,
                                                opti_method = self.opti_method,
                                                opti_params = self.opti_params,
                                                clustering_params = self.clustering_params)

        # self =AddFeaturesClusterWindow2(X = X_window,
        #                                 date = date,
        #                                 clust_algo = self.clust_algo,
        #                                 metric = self.metric,
        #                                 opti = self.opti,
        #                                 opti_method = self.opti_method,
        #                                 opti_params = self.opti_params,
        #                                 clustering_params = self.clustering_params)

        # self.labels = labels
        # self.unique_labels = np.unique(labels)
        # self.nb_labels = len(set(labels))
        # self.res
        
        # self.labels.shape
        # X_window.shape
        # self.get_features(refit=False)

        labels.shape
        current_model.labels = labels
        # self.prev_model.labels.shape
        # prev_labels.shape
        # prev_X.shape
        # labels.shape
        # X_window.shape
        # current_X.shape
        # current_model.labels
        current_model.unique_labels = np.unique(labels)
        current_model.nb_labels = len(set(labels))
        try :
            current_model.get_features(refit=False)
        except : 
            print(f'last labels tot : {self.prev_model.labels.shape}')
            print(f'X_window : {X_window.shape}')
            print(f'labels {labels.shape}')
            print(f'prev_X :{prev_X.shape}')
            print(f'prev_label : {prev_labels.shape}')
            print(f'current_X {current_X.shape}')
            raise ValueError('Pb with get_features')
        self.last_condition
        return current_model, current_model.res

    def clusterisation_with_refit(self,df_window,X_window,date) : 
        current_model = AddFeaturesClusterWindow2(df_window=df_window,
                                                X = X_window,
                                                date = date,
                                                clust_algo = self.clust_algo,
                                                metric = self.metric,
                                                opti = self.opti,
                                                opti_method = self.opti_method,
                                                opti_params = self.opti_params,
                                                clustering_params = self.clustering_params)
        # self = AddFeaturesClusterWindow2(df_window=df_window,
        #                                         X = X_window,
        #                                         date = date,
        #                                         clust_algo = self.clust_algo,
        #                                         metric = self.metric,
        #                                         opti = self.opti,
        #                                         opti_method = self.opti_method,
        #                                         opti_params = self.opti_params,
        #                                         clustering_params = self.clustering_params)

        # self.get_features(refit=True)
        current_model.get_features(refit=True)
        return current_model,current_model.res

    def run(self) : 
        all_dates = self.df_embedding['date'].drop_duplicates().sort_values()
        all_dates = all_dates[all_dates > all_dates.iloc[0] + pd.Timedelta(days=self.lookback_window)]
        for date in tqdm(all_dates):
            ## df_window
            # date = all_dates.iloc[8]
            # date = pd.to_datetime('2018-11-24 13:06:00+00:00')
            # break
            start_date = date - pd.Timedelta(days=self.lookback_window)
            mask = ((self.df_embedding['date'] > start_date) & (self.df_embedding['date'] <= date - self.max_forward_looking )) | (self.df_embedding['date'] == date)
            df_window = self.df_embedding.loc[mask]
            X_window = self.get_X(df_window)
            ## Skip the date if the nb of news is less than the the min_nb_news (not enough sample)
            if len(df_window) < self.min_nb_news :  
                self.prev_date = None
                self.last_condition = 1
                continue  
            
            ## Initialisation
            ### Pour DBSCAN
            elif self.prev_date == None : 
                self.prev_model, res = self.clusterisation_with_refit(df_window,X_window,date)
                self.last_condition = 2
                self.prev_X_window = X_window

            elif self.prev_date.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d') : ## au jour pret
                try : 
                    self.prev_model, res = self.clusterisation_without_refit(df_window,X_window,date)
                    self.prev_X_window = X_window
                    self.last_condition = 3
                except : 
                    print(self.prev_date)
                    print(len(self.prev_X_window))
                    print(date)
                    print(len(X_window))
                    raise ValueError()
                    break
                    # raise ValueError
            else : 
                self.prev_model, res = self.clusterisation_with_refit(df_window,X_window,date)
                self.prev_X_window = X_window
                self.last_condition = 4
            
            self.prev_df_window = df_window.copy()
            self.prev_date = date
            # X_window.shape
            try :
                res.loc[:,'idx_news'] = df_window.iloc[-1]['idx_news']
            except : 
                raise ValueError('The idx doesnt work')
            self.df_res = pd.concat([self.df_res, res])
        
        self.df_res.columns =[f'feat_{c}' for c in self.df_res.columns if c != 'idx_news'] + ['idx_news']
        self.df_res.reset_index(inplace=True)
        return self.df_res


def merge_features_before_clusterisation(df_sent,df_emb,df_ohlc,n_minutes=[5,15,30]) :
    col = 'title_desc' 
    common_col = ['date', 'url', 'author', 'title', 'subtitle', 'source', 'title_desc','idx_news']
    sent_col = [f'clean_sent_{col}']  ## a changer en clean_sent
    emb_col = [f'clean_emb_{col}']
    feat_emb = [c for c in df_emb.columns if c.startswith('feat_emb')]          ## a changer en feat emb
    feat_sent = [c for c in df_sent.columns if c.startswith('feat_sent')]          ## a changer en feat emb
    assert (df_sent[common_col].values == df_emb[common_col].values).all(), 'not same value for the common columns'
    df = pd.merge(df_sent[common_col + sent_col], df_emb[common_col + emb_col],on = common_col,how = 'inner')
    df = pd.concat([df,df_sent[feat_sent],df_emb[feat_emb]],axis=1)
    df = add_forward_return(df,df_ohlc,n_minutes,col_name='feat_ret')
    df.date.is_monotonic_increasing
    # df.iloc[1000:1100]
    return df

def get_clusters(df,
                 emb_model,
                 clust_algo,
                 opti=True,
                 opti_method=None,
                 opti_params=None,
                 clustering_params = None,
                 lookback_window = 7,
                 min_nb_news = 10,
                 max_forward_looking = 30) :
   
    self =  AddFeaturesCluster2(df_embedding=df,
                        emb_model=emb_model,
                        clust_algo=clust_algo,
                        metric='cosine',
                        opti = opti,
                        opti_method=opti_method,
                        opti_params= opti_params,
                        clustering_params= clustering_params,
                        lookback_window = lookback_window,
                        min_nb_news=min_nb_news,
                        max_forward_looking= max_forward_looking)
    df_feat = self.run()

def get_features(filename,sent_model,emb_model,df_ohlc,add_cluster_args,n_minutes=[5,15,30]) :
    ## Embedding
    path_emb = './data/embedding/'
    filename_emb = f'{filename}_emb_{emb_model}.csv'
    df_emb = get_data(path = path_emb + filename_emb, filetype = 'csv')
    df_emb = to_datetime(df_emb,col='date',round=False)
    ## Sentiment
    path_sent = './data/sentiment/'
    filename_sent = f'{filename}_sent_{sent_model}.csv'
    df_sent = get_data(path = path_sent + filename_sent, filetype = 'csv')
    df_sent = to_datetime(df_sent,col='date',round=False)

    test_rigth_dataframe(df_sent,df_emb)
    df = merge_features_before_clusterisation(df_sent,df_emb,df_ohlc,n_minutes=n_minutes)
    # add_feat_clust =  AddFeaturesCluster2(df_embedding=df,
    #                                       emb_model=emb_model,
    #                                       clust_algo=clust_algo,
    #                                       metric='cosine',
    #                                       opti = opti,
    #                                       opti_method=opti_method,
    #                                       opti_params= opti_params,
    #                                       clustering_params= clustering_params,
    #                                       lookback_window = lookback_window,
    #                                       min_nb_news=min_nb_news,
    #                                       max_forward_looking= max_forward_looking)
    add_feat_clust = AddFeaturesCluster2(df,emb_model,**add_cluster_args)
    df_feat = add_feat_clust.run()
    
    path_feat = './data/features/'
    clust_algo = add_cluster_args.get('clust_algo',False)
    opti_method = add_cluster_args.get('opti_method',False)
    if  add_cluster_args.get('opti',False) :    
        filename_feat = f'{filename}_feat_{sent_model}_{emb_model}_{clust_algo}_{opti_method}'
    else :
        filename_feat = f'{filename}_feat_{sent_model}_{emb_model}_{clust_algo}'

    df_save_data(df_feat,path_feat,filename_feat,'csv',create_folder=True)



if __name__ == '__main__' :
    ## params
    sent_model = 'Vader' 
    emb_model = 'word2vec'

    n_minutes = [5,15,30]
    add_clust_args = dict (clust_algo = 'dbscan',
                         opti = True,
                         opti_method = 'nb_cluster',
                         opti_params = dict(range_eps = np.arange(0.01,1,0.03)
                                            ,size_min=5),
                        lookback_window= 7,
                        min_nb_news = 20,
                        clustering_params = dict(size_min = 5),
                        max_forward_looking = 30)

    ## Market data
    path = './data/market_data/'
    filename = 'df_btc_1m_2014-12-01_2025-07-15.json'
    df_ohlc = pd.read_json(path+filename)
    df_ohlc = to_datetime(df_ohlc,col='date',round=False)

    get_features('df_cointelegraph',sent_model,emb_model,df_ohlc,add_clust_args,n_minutes)

    

    




