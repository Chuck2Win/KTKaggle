! pip install --upgrade pip
! pip install catboost

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,classification_report
from catboost import CatBoostClassifier
import pickle
from sklearn.impute import SimpleImputer
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline#FeatureUnion,
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm,Pool
from sklearn.model_selection import train_test_split

data = pd.read_csv('merged_data.csv')
simpleimputer=SimpleImputer(missing_values = None, strategy = 'most_frequent')
data['age_itg_cd'][(data['age_itg_cd']=='_')]=None
data['age_itg_cd']=simpleimputer.fit_transform(data['age_itg_cd'].values.reshape(-1,1))[:,0]
data['age_itg_cd']=data['age_itg_cd'].astype(int)
data = data.drop('id',axis=1)
y = data[['label_payment_yn']]
X = data.drop(['label_payment_yn','phone_exists'],axis=1)   
f = open('categorical_features','rb')
categorical_features = pickle.load(f)

# grid search
model = CatBoostClassifier(iterations=10, use_best_model=False, eval_metric='Accuracy',) # one hot max size, iteration
grid = {'one_hot_max_size': [1, 3, 5, 7, 9, 12],} # one hot을 하는 max size
grid_search_result = model.grid_search(grid, 
                                       X=Pool(X,cat_features=categorical_features,label=y), verbose = 10                             
                                       
                                   )

# 이상치 제거 x
model = CatBoostClassifier(one_hot_max_size=9, iterations=100, use_best_model=True, eval_metric='Accuracy',) # one hot max size, iteration
model.fit(pd.concat([X_train,X_test]),pd.concat([y_train,y_test]),cat_features = categorical_features,eval_set=(X_test,y_test),verbose=50,use_best_model=True)

# feature selection RFE 방식
model2 = CatBoostClassifier(one_hot_max_size=9, iterations=100, use_best_model=True, eval_metric='Accuracy') # one hot max size, iteration
summary2 = model2.select_features(
    Pool(X_train,label=y_train,cat_features = categorical_features),
    eval_set=Pool(X_test,label=y_test,cat_features = categorical_features),
    features_for_select=list(X_train.columns),
    num_features_to_select=55, # hyperpara임
    steps=2, # 얼마나 학습을 시킬 것인가?
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=False,
    verbose=50)
