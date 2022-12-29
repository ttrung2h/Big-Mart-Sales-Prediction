import pandas as pd
import numpy as np

class BigMartSalesDataPipeLine :
    def __init__(self):
        self.df = None
    def fit(self,root_train :str,root_test : str):
        '''
            Fit dir into pipeline
        '''
        train_df = pd.read_csv(root_train)
        test_df = pd.read_csv(root_test)
        
        # Create new feature :source
        train_df['source'] = 'train'
        test_df['source'] = 'test'
        
        # set value target in test df = 0
        test_df['Item_Outlet_Sales'] = 0
        self.df = pd.concat([train_df,test_df],sort=False,ignore_index=True)
    def transforms(self):
        '''
            Transform data and return dataframe after processing
        '''
        self.processing_inconsistent()
        self.processing_missing_value()
        self.processing_outlier()
        self.processing_skew()
        return self.df
    
    def fit_transforms(self,root_train:str,root_test : str):
        '''
            Include step train and test
        '''
        self.fit(root_train=root_train,root_test=root_test)
        self.transforms()
        return self.df
    def processing_inconsistent(self):
        '''
            Function will process consistent in 'Item Weight'
        '''
        feature = 'Item_Fat_Content'
        self.df[feature].replace(to_replace = 'LF',value = 'Low Fat',inplace=True)
        self.df[feature].replace(to_replace= 'low fat',value = 'Low Fat',inplace=True)
        self.df[feature].replace(to_replace= 'reg',value = 'Regular',inplace = True)
    def processing_missing_value(self):
        '''
            Processing null value in feature
        '''
        
        # Processing null in `Item _Weight`
        null_feature_1 = 'Item_Weight'
        pivot_feature_1 = 'Item_Identifier'
        table_weight_itemWeight = self.df.pivot_table(values= null_feature_1,index= pivot_feature_1, aggfunc = 'mean')
        index_null_itemweight = list(self.df[self.df[null_feature_1].isnull() == True].index)
        replace_series_itemweight ={idx:table_weight_itemWeight.loc[self.df.loc[idx,pivot_feature_1],null_feature_1] for idx in index_null_itemweight}
        # Replace null value
        self.df[null_feature_1] = self.df[null_feature_1].fillna(replace_series_itemweight)
        
        
        # Processing null in`Outlet_Size`
        null_feature_2 = 'Outlet_Size'
        self.df[null_feature_2].replace(np.NaN,'Unknown',inplace=True)
        pivot_feature_2 = "Outlet_Type"
        self.df[null_feature_2].replace("Unknown","Small",inplace = True)
        
        
        # Processing null in 'Item_Visibility': value equals 0 => null value
        null_feature_3 = 'Item_Visibility'
        self.df[null_feature_3].replace(to_replace=0,value=np.nan,inplace=True)
        feature_cmp_1 = "Item_Type"
        feature_cmp_2 = "Outlet_Type"
        table_weight = self.df.pivot_table(values = null_feature_3, index = feature_cmp_1, columns = feature_cmp_2, aggfunc = "mean")
        index_null = list(self.df[self.df[null_feature_3].isnull()].index)
        replace_Nan = {idx : table_weight.loc[self.df.loc[idx,feature_cmp_1],self.df.loc[idx,feature_cmp_2]] for idx in index_null}
        # Impute NaN
        self.df[null_feature_3].fillna(replace_Nan,inplace=True)
    
    def processing_outlier(self):
        '''
            In this function we will delete outlier because number of outlier value is small
        '''   
        feature = "Item_Visibility"
        q1,q2,q3 =np.quantile(self.df[feature].values,0.25),np.quantile(self.df[feature].values,0.5),np.quantile(self.df[feature].values,0.75)
        iqr = q3 - q1
        lower_bound,upper_bound = q1 - 1.5*iqr,q3 + 1.5*iqr
        # Getting outlier data
        outlier_data = self.df[(self.df[feature] < lower_bound) | (self.df[feature]> upper_bound)]
        
        # Only drop data in source train
        outlier_index_train = outlier_data[outlier_data['source'] == 'train'].index
        self.df = self.df.drop(outlier_index_train)
    
    def processing_skew(self):
        '''
            Processing skew of data in feature `Item_Visibility`
        '''
        feature = 'Item_Visibility'
        self.df[self.df['source'] == 'train'][feature] = np.sqrt(self.df[self.df['source'] == 'train'][feature])