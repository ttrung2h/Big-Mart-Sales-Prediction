import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

class BigMartSaleModelPipeLine:
    def __init__(self) -> None:
        pass
    
    def fit(self,X:pd.DataFrame,y:pd.DataFrame,best_params_file = None):
        '''
            Fit data for training model
        '''
        self.X = self.delete_digit_after_id(self.create_age_feature(X))
        self.y = y.values.ravel()
        self.best_params_file = best_params_file
        self.create_pipeline()
   
    def create_pipeline(self):
        '''
            Create model pipeline include : {Standarscaler,LabelEncoder,RandomForestRegressor}
        '''
        self.dict_encode = self.encode_data()
        model = RandomForestRegressor()
        if self.best_params_file :
            with open(self.best_params_file, 'r', encoding='utf-8') as f:
               best_parameter = json.loads(f.read())
            model.set_params(**best_parameter) 

        scaler = StandardScaler()
        self.Pipeline = Pipeline([
            ('scaler',scaler),
            ('model',model)
        ]).fit(self.X,self.y)    
        
        print(self.Pipeline)
    def encode_data(self):
        '''
            Encoding data and save the encode objectin dict
        '''
        dict_encode = {}
        category_col = [col for col in self.X.columns if self.X[col].dtype == 'O']
        for col in category_col:
            encode = LabelEncoder()
            self.X[col] = encode.fit_transform(self.X[[col]])
            dict_encode[col] = encode
        return dict_encode
    
    def create_age_feature(self,X):
        '''
            Create feature Age base on `Outlet_Establishment_Year`
        ''' 
        cur_year = datetime.now().year
        X['Age'] = cur_year - X['Outlet_Establishment_Year']
        X.drop(columns=['Outlet_Establishment_Year'],inplace=True)
        return X
   
    def delete_digit_after_id(self,X):
        '''
            Delete digit after id 
            Example FD123 => FD
        '''
        feature = 'Item_Identifier'
        X[feature] = X[feature].apply(lambda x : x[:-2])
        return X
    def predict(self,X_test: pd.DataFrame):
        '''Create predict to return prediction of model'''
        X_test = self.create_age_feature(X_test)
        X_test = self.delete_digit_after_id(X_test)
        for col in self.dict_encode.keys():
            X_test[col] = self.dict_encode[col].transform(X_test[[col]])
        return self.Pipeline.predict(X_test)
   