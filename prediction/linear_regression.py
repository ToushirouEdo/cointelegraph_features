import importlib 
import utils.processing 
importlib.reload(utils.processing)
from utils.processing import get_data, df_save_data,to_datetime


import utils.market_data 
importlib.reload(utils.market_data )
from utils.market_data  import add_forward_return

import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm


class LinearRegression :
    def __init__(self,df,penalization = None) : 
        self.df = df.copy()
        self.penalization = penalization

        self.rescale = True if penalization != None else False

    def processing(self,y_col) : 
        self.df.dropna(subset=[c for c in self.df.columns if c.startswith('target')],inplace = True)
        self.df.set_index('date',inplace=True)
        self.df = self.df.astype(float)
        self.df = self.df.fillna(0)
       
        
        X = self.df[[c for c in self.df.columns if c.startswith('feat')]]
        y = self.df[y_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, shuffle = False)
        self.X_val, self.X_test, self.y_val, y_test = train_test_split(self.self.X_test,self.y_test,test_size=0.5,shuffle=False)
        
    def get_lassoCV(self) :
        if self.rescale :
            scaler = StandardScaler().fit(self.X_train)
            self.X_train_scaled = scaler.transform(self.X_train)
            self.X_test_scaled = scaler.transform(self.X_test)
        
    
        # Fit LassoCV (automatically finds the best alpha)
        lasso_cv = LassoCV(cv=5, max_iter=5000)
        lasso_cv.fit(self.X_train_scaled, self.y_train)

        # Predictions and evaluation
        y_pred = lasso_cv.predict(self.X_test_scaled)
        lasso_cv.coef_
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"[LassoCV] Best alpha: {lasso_cv.alpha_}")
        print(f"[LassoCV] MSE: {mse:.4f}")
        print(f"[LassoCV] R²: {r2:.4f}")
        print("[LassoCV] Coefficients:")
        print(pd.Series(lasso_cv.coef_, index=self.X_train.columns))
        return lasso_cv
        
    def get_OLS(self,statsmodel=True) :
        if not statsmodel:
            raise NotImplementedError("Only statsmodels OLS is implemented here.")

        X_train_ols = sm.add_constant(self.X_train)
        X_test_ols = sm.add_constant(self.X_test)

        model = sm.OLS(self.y_train, X_train_ols)
        result = model.fit()

        print(result.summary())

        y_pred = result.predict(X_test_ols)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"[OLS] MSE: {mse:.4f}")
        print(f"[OLS] R²: {r2:.4f}")

        return result






if __name__ == '__main__' :
    
    ## Features
    path_feat = './data/features/'
    filename = 'df_cointelegraph_feat_Vader_word2vec_dbscan_nb_cluster.csv'
    df_feat = get_data(path = path_feat + filename, filetype = 'csv')
    df_feat= to_datetime(df_feat,col='date',round=False)

     
    

    ## Market data
    path = './data/market_data/'
    filename = 'df_btc_1m_2014-12-01_2025-07-15.json'
    df_ohlc = pd.read_json(path+filename)
    df_ohlc = to_datetime(df_ohlc,col='date',round=False)
    
    df = add_forward_return(df_feat,df_ohlc,n_minutes=[15,30,60,90,180],col_name='target')
    
    