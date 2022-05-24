import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from conda_env.installers import conda, pip
from conda_env.installers.pip import install
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from helpers.eda import *
from helpers.helpers import *
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pip install xgboost
pip install catboost
pip install lightgbm
# conda install -c conda-forge lightgbm
import catboost
import lightgbm
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
df = pd.read_csv("https://www.kaggle.com/datasets/floser/hitters?select=Hitters.csv")
df_hitters=df.copy()
df.head()

# QUICK DATA-PREP
df_hitters.dropna(inplace=True)
cat_cols,cat_but_car,num_cols,num_but_cat=grab_col_names(df_hitters)

for col in cat_cols:
    cat_summary(df_hitters,col)

for col in cat_cols:
    label_encoder(df_hitters,col)

for col in num_cols:
    if 'Salary' not in col:
        transformer=RobustScaler().fit(df_hitters[[col]])
        df_hitters[col]=transformer.transform(df_hitters[[col]])

# another alternative functional approach
for col in [col for col in num_cols if "Salary" not in col]:
    transformer = RobustScaler().fit(df_hitters[[col]])
    df_hitters[col] = transformer.transform(df_hitters[[col]])

y=df_hitters["Salary"]
X=df_hitters.drop("Salary",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=17)

models=[("LR",LinearRegression()),
        ("Ridge",Ridge()),
        ("Lasso",Lasso()),
        ("ElasticNet",ElasticNet()),
        ("KNN",KNeighborsRegressor()),
        ("CART",DecisionTreeRegressor()),
        ("RF",RandomForestRegressor()),
        ("SVR",SVR()),
        ("GBM",GradientBoostingRegressor()),
        ("XGBoost",XGBRegressor()),
        ("LightGBM",LGBMRegressor()),
        ("CatBoost",CatBoostRegressor(verbose=False))
        ]

# Test errors for base models
for name,model in models:
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    msg="%s: (%f)" % (name,rmse)
    print(msg)

# Examination of base models by applying CV to all data
for name, model in models:
    rmse=np.mean(np.sqrt(-cross_val_score(model,X,y,cv=5,scoring="neg_mean_squared_error")))
    msg="%s: (%f)" % (name,rmse)
    print(msg)


# Examination of base models by applying  HOLDOUT +  CV to all data
for name, model in models:
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    rmse=np.mean(np.sqrt(-cross_val_score(model,X_test,y_test,cv=5,scoring="neg_mean_squared_error")))
    msg="%s: (%f)" % (name,rmse)
    print(msg)


for i in models:
    combns[i]=list(itertools.combinations(models,2))
    combns[i][0] and combns[i][1]

from sklearn.ensemble import VotingRegressor

# voting for each model by one by;
for name,model in models:
    voting_reg=VotingRegressor(estimators=models,n_jobs=-1,verbose=2).fit(X_train,y_train)
    y_pred=voting_reg.predict(X_test)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    msg = "%s: (%f)" % (name, rmse)
    print(msg)

#voting for each model by combination
import itertools
combn = list(itertools.combinations(models, 2))
# another alternative functional approach;
combn[0]
group_list = []
for i in itertools.combinations(models, 2):
    group_list.append(i)

# voting for each model by combination with Hold-out approach
for model in combn:
    voting_reg=VotingRegressor(estimators=model,n_jobs=-1,verbose=2).fit(X_train,y_train)
    y_pred = voting_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = " (%f)" % (rmse)
    print(msg)

# voting for each model by combination with Hold-out + CV approach
for model in combn:
    voting_reg = VotingRegressor(estimators=model, n_jobs=-1, verbose=1).fit(X_train, y_train)
    cross_val_score(voting_reg, X_train, y_train, cv=5).mean()
    y_pred = voting_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = " (%f)" % (rmse)
    print(msg)


