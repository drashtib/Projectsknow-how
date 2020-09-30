# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:30:43 2020

@author: Drashti Bhatt
"""
import numpy as np # linear algebra
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Muting warning messages
import warnings
warnings.filterwarnings('ignore')
#read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# Applying a log transformation in the column "SalePrice"
train["SalePrice"] = np.log(train["SalePrice"])

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

#Plotting the Distribution and the QQPlot
plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.distplot(train['SalePrice'], fit=norm)
g.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='top-right')

plt.subplot(1, 2, 2)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

corr = train.corr()
plt.subplots(figsize=(30, 30))
cmap = sns.diverging_palette(10, 150, n=9, as_cmap=True, center="light")
sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1.0, center=0.2, square=True, linewidths=0, cbar_kws={"shrink": .5}, annot = True);

# Checking for outliers using the 'GrLivArea' variable
sns.lmplot('GrLivArea', 'SalePrice', train, size=5, aspect=2)

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<250000)].index)

sns.lmplot('GrLivArea', 'SalePrice', train, size=5, aspect=2)

# Saving each datasets length, in order to split them again later.
train_length = len(train)
test_length = len(test)

# Saving the 'SalePrice' column that is only included in the Train Dataset. We will remove it and append it again later.
y_train = train.SalePrice.values

# Concatenating the datasets
joint = pd.concat((train, test)).reset_index(drop=True)

# Dropping the 'SalePrice' column, because it has values only for the train dataset
joint.drop(['SalePrice'], axis=1, inplace=True)

# Sum all the missing values
NAs = joint.isnull().sum()

# Filtering out the columns that don't have missing values
NAs = NAs.drop(NAs[NAs == 0].index).sort_values(ascending=False)

# Plotting the count bars to get an idea of the missing values each column has
NAs.plot(kind='bar', figsize =(17, 5))

col = ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", 
       "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", 
       "MasVnrType", "MSSubClass")

for i in col:
    joint[i] = joint[i].fillna("None")
    
col = ("GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", 
       "TotalBsmtSF", "MasVnrArea", "BsmtFullBath", "BsmtHalfBath")

for i in col:
    joint[i] = joint[i].fillna(0)

col = ("MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType", "Functional")

for i in col:
    joint[i] = joint[i].fillna(joint[i].mode()[0])

# Fixing missing values for GarageYrBlt
joint["GarageYrBlt"] = joint["GarageYrBlt"].fillna(joint["YearBuilt"])

# Fixing missing values for LotFrontage
joint["LotFrontage"] = joint.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# Dropping the Utilities variable
joint = joint.drop(['Utilities'], axis=1)

# Total Sq Feet of the house
joint['TotalSF'] = joint['TotalBsmtSF'] + joint['1stFlrSF'] + joint['2ndFlrSF']

# Freshness: How old was the house when it was sold
joint['Freshness'] = joint['YrSold'] - joint['YearBuilt']

# Converting to categorical features
col = ("YrSold", "MoSold", "OverallCond")
for i in col:
    joint[i] = joint[i].astype(str)
    
joint['MSSubClass'] = joint['MSSubClass'].apply(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for i in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(joint[i].values)) 
    joint[i] = lbl.transform(list(joint[i].values))

# shape        
print('Shape joint: {}'.format(joint.shape))

# Quadratic Transformation for the top numeric variables
joint["OverallQual^2"] = joint["OverallQual"] ** 2
joint["GrLivArea^2"] = joint["GrLivArea"] ** 2
joint["GarageCars^2"] = joint["GarageCars"] ** 2
joint["GarageArea^2"] = joint["GarageArea"] ** 2
joint["TotalBsmtSF^2"] = joint["TotalBsmtSF"] ** 2
joint["1stFlrSF^2"] = joint["1stFlrSF"] ** 2
joint["FullBath^2"] = joint["FullBath"] ** 2
joint["TotRmsAbvGrd^2"] = joint["TotRmsAbvGrd"] ** 2

# Cubic Transformation for the top numeric variables
joint["OverallQual^3"] = joint["OverallQual"] ** 3
joint["GrLivArea^3"] = joint["GrLivArea"] ** 3
joint["GarageCars^3"] = joint["GarageCars"] ** 3
joint["GarageArea^3"] = joint["GarageArea"] ** 3
joint["TotalBsmtSF^3"] = joint["TotalBsmtSF"] ** 3
joint["1stFlrSF^3"] = joint["1stFlrSF"] ** 3
joint["FullBath^3"] = joint["FullBath"] ** 3
joint["TotRmsAbvGrd^3"] = joint["TotRmsAbvGrd"] ** 3


# Square Root Transformation for the top numeric variables
joint["OverallQual-Sq"] = np.sqrt(joint["OverallQual"])
joint["GrLivArea-Sq"] = np.sqrt(joint["GrLivArea"])
joint["GarageCars-Sq"] = np.sqrt(joint["GarageCars"])
joint["GarageArea-Sq"] = np.sqrt(joint["GarageArea"])
joint["TotalBsmtSF-Sq"] = np.sqrt(joint["TotalBsmtSF"])
joint["1stFlrSF-Sq"] = np.sqrt(joint["1stFlrSF"])
joint["FullBath-Sq"] = np.sqrt(joint["FullBath"])
joint["TotRmsAbvGrd-Sq"] = np.sqrt(joint["TotRmsAbvGrd"])
#Treating skewed features
# Getting the indices of numeric columns
numcolumns = joint.dtypes[joint.dtypes != "object"].index

# Check how skewed they are
skewed = joint[numcolumns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

plt.subplots(figsize =(17, 8))
skewed.plot(kind='bar');

skewness = skewed[abs(skewed) > 0.75]
from scipy.special import boxcox1p
skewed = skewness.index
lam = 0.15
for i in skewed:
    joint[i] = boxcox1p(joint[i], lam)

print(skewness.shape[0],  "skewed numerical features have been Box-Cox transformed")

joint = pd.get_dummies(joint)
train = joint[:train_length]
test = joint[train_length:]

n_folds = 10

def rmsle_cv(model):
    kfolds = KFold(n_folds, shuffle=True, random_state=13).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kfolds))
    return(rmse)

#lasso
lasso = Lasso(alpha =0.0005, random_state=1)
lasso.fit(train.values, y_train)
lasso_pred = np.exp(lasso.predict(test.values))
score = rmsle_cv(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

lasso1 = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso1.fit(train.values, y_train)
lasso1_pred = np.exp(lasso1.predict(test.values))
score = rmsle_cv(lasso1)
print("Lasso (Robust Scaled) score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

##elastic net models
elnet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=42)
elnet.fit(train.values, y_train)
elnet_pred = np.exp(elnet.predict(test.values))
score = rmsle_cv(elnet)
print("Elastic Net score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from sklearn.tree import DecisionTreeRegressor

all_depths = []
all_mean_scores = []
for max_depth in range(1, 11):
    all_depths.append(max_depth)
    simple_tree = DecisionTreeRegressor(max_depth=max_depth)
    cv = KFold(n_splits=5, shuffle=True, random_state=13)
    scores = cross_val_score(simple_tree, train.values, y_train, cv=cv)
    mean_score = np.mean(scores)
    all_mean_scores.append(np.mean(scores))
    print("max_depth = ", max_depth, scores, mean_score, sem(scores))

# Designing the Decision Tree Regressor with max_depth = 5
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(train.values, y_train)

# Extracting the predictions for the test dataset
dtree_pred = np.exp(tree.predict(test.values))

# Checking the RMSE Score
score = rmsle_cv(tree)
print("Decision Tree score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

from sklearn.ensemble import RandomForestRegressor

# Building the Random Forest Regressor
rfr = RandomForestRegressor(max_depth=None, random_state=0, min_samples_split=2, 
                              n_estimators=100)
rfr.fit(train.values, y_train)
rfr_pred = np.exp(rfr.predict(test.values))
score = rmsle_cv(rfr)
print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(tree, n_estimators=500)
ada.fit(train.values, y_train)
ada_pred = np.exp(ada.predict(test.values))
score = rmsle_cv(ada)
print("Ada Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

xg_reg = xgb.XGBRegressor(objective ='reg:linear', 
                          colsample_bytree = 0.5, 
                          learning_rate = 0.15,
                          max_depth = 4, 
                          reg_lambda = 0.5, 
                          n_estimators = 100)
xg_reg.fit(train.values, y_train)
xg_reg_pred = np.exp(xg_reg.predict(test.values))
score = rmsle_cv(xg_reg)
print("XGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

from sklearn.model_selection import GridSearchCV

# We create a parameter grid
params = {
    'objective': ['reg:linear'],
    'colsample_bytree': [0.4, 0.5, 0.6],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5], 
    'reg_lambda': [0.5, 0,75, 1],
    'reg_alpha': [0.2, 0.4, 0.6],
    'subsample': [0.25, 0.5]
}



xgb_model = xgb.XGBRegressor()

# We use Grid Search in order to run 5-fold cross-validation and select the optimal values for the parameters
clf = GridSearchCV(xgb_model, 
                   params, 
                   n_jobs=-1, 
                   cv=5, 
                   scoring='neg_mean_absolute_error',
                   verbose=2,
                   iid=False,
                   refit=True)

clf.fit(train.values, y_train)
clf.best_estimator_
xg_reg = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
xg_reg.fit(train.values, y_train)
xg_reg_pred = np.exp(xg_reg.predict(test.values))
score = rmsle_cv(xg_reg)
print("XGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#ANN
import tensorflow as tf
from tensorflow import keras
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu',
                                 input_shape=(train.shape[1],)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model

skf = KFold(n_splits=5)
num_epochs = 100
all_scores = []
all_mse_histories = []

for i, (train_indx, test_indx) in enumerate(skf.split(train.values, y_train)):
    print(f'Processing Fold: {i+1}')
    partial_train_data = train.values[train_indx]
    partial_train_targets = y_train[train_indx]
    partial_val_data = train.values[test_indx]
    partial_val_targets = y_train[test_indx]
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, 
                        validation_data=(partial_val_data, partial_val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mse_history = history.history['mean_squared_error']
    all_mse_histories.append(mse_history)
    
print('Done')