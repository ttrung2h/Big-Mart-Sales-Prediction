import pandas as pd
from DataPipeLine.data_pipe_line import BigMartSalesDataPipeLine
from Model_Pipe_Line.model_pipe_line import BigMartSaleModelPipeLine
import json
import warnings
import pickle
warnings.filterwarnings("ignore")
def main():
    root_train = 'T:\SalePrediction\data\sale_data\BigMartSales Prediction\Train.csv'
    root_test = 'T:\SalePrediction\data\sale_data\BigMartSales Prediction\Test.csv'
    data_pipe_line = BigMartSalesDataPipeLine()
    df = data_pipe_line.fit_transforms(root_train,root_test)
    # Get X ,y for train 
    target = 'Item_Outlet_Sales'  
    X_train = df[df['source'] == 'train'].drop(columns=[target,'source'])
    y_train = df.loc[X_train.index,target]
    X_test = df[df['source'] == 'test'].drop(columns=[target,'source'])
    
    model = BigMartSaleModelPipeLine()
    model.fit(X_train,y_train,best_params_file='T:\\SalePrediction\\best_prams.json')

    final_result = pd.read_csv(root_test)
    final_result[target] = model.predict(X_test=X_test) 
    #Save result 
    pickle.dump(model, open("Model/BigMartSaleModel.pkl", "wb"))
    final_result.to_csv('Prediction/result.csv')
if __name__ == '__main__':
    main()
    
     
    