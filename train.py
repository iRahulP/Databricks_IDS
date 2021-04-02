import os
import warnings
import sys

import pandas as pd
import seaborn as sns
import re
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,recall_score,precision_score,f1_score
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    ids_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IDS_train.csv")
    data = pd.read_csv(ids_path)

    #Stores Pandas dataframe in Train
    Train = data
    
    #refactoring nulls
    Train["Flow Bytes/s"]=Train["Flow Bytes/s"].fillna(0)
    
    #Combining different types of attacks as one using list comprehension according to problem statement
    Target_1=["DoS Slowhttptest","DoS Hulk","DoS GoldenEye","DoS slowloris"]
    Train["Label"] = [1 if X in(Target_1) else 0 for X in Train["Label"]]
    
    #Replaced the infinity values with NA'S and removed them.
    Train=Train[Train.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    Train.dropna(axis=0,inplace=True)
    
    #To make all the columns uniform, changing it to one of the dtpyes i.e int to float.
    int_cols_n = list(Train.select_dtypes(include='int64').columns)
    for i in int_cols_n:
        Train[i].astype('float64')
    
    Train['Label']=Train['Label'].astype('category')
    Train=Train.drop(columns=['ID'])

    # get all rows and columns but the last column, which is our class
    X = Train.drop(columns='Label')
    # get all observed values in the last columns, which is what we want to predict
    y = Train.loc[:,'Label']

    # create train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #estimators = int(sys.argv[1])    
    estimators = 10    
    

    with mlflow.start_run():
        # train and predict
        lr = RandomForestClassifier(n_estimators = estimators)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        
        # compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test,y_pred)
        
        print("Random Forest Classifier model (n_estimators=%d):" % (estimators))
        print("  accuracy: %s" % acc)
        print("  precision: %s" % precision)
        print("  confusion_matrix: %s" % conf_matrix)
        
        # mlflow.log_metric("accuracy_score: %s" % acc)
        # mlflow.log_metric("precision: %s" % precision)
        # mlflow.log_metric("confusion_matrix: %s" % conf_matrix)
        
        mlflow.sklearn.log_model(lr, "model")