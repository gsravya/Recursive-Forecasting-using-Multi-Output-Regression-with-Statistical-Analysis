#-----------------
#    IMPORTS
#-----------------
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import GridSearchCV, ShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFECV, SelectFromModel, f_regression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from math import sqrt
from sklearn.feature_selection import RFE
from matplotlib.pylab import rcParams
from statsmodels.formula.api import ols
from statsmodels.stats import anova
from sklearn.preprocessing import scale

#---------------------
# Function Definition
#---------------------
    
def iter_feature(val):
    #print val
    new_arr = val[1:-1]
    #print new_arr
    return new_arr

#------------------    
#other declarations
#------------------
rcParams['figure.figsize'] = 15, 6
np.random.seed(0)
forecast_labels = []
actual_labels = []
forecast_rmse = []
actual_days = ["2016-06-05","2016-06-10","2016-06-15","2016-06-18","2016-06-20","2016-06-25","2016-06-28","2016-07-05","2016-07-10","2016-07-15","2016-07-18","2016-07-20","2016-07-25","2016-07-28"]
forecaste_days = ["2016-08-05","2016-08-10","2016-08-15","2016-08-18","2016-08-20","2016-08-25","2016-08-28","2016-09-05","2016-09-10","2016-09-15","2016-09-18","2016-09-20","2016-09-25","2016-09-28"]


#------------------------
#MAIN Program code starts
#------------------------
#load housing data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('../../Data/MA_dataset.csv', parse_dates=[0], index_col='DT',date_parser=dateparse)
X = data.values


#-------------------------------
#PLOTS -For statistical Analysis
#-------------------------------
pd.tools.plotting.scatter_matrix(data.loc[:,"MLP":"MSP"],diagonal="kde")
plt.tight_layout()
plt.savefig("../Figures/MA/matrix_scatter.png")
plt.clf()

sns.lmplot("MSP", "MLP", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs Median Listing Price")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_MLP.png")
plt.clf()

sns.lmplot("MSP", "MPC", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs Median Price Cut")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_MPC.png")
plt.clf()

sns.lmplot("MSP", "SFL", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs %Sold For Loss")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_SFL.png")
plt.clf()

sns.lmplot("MSP", "SFG", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs %Sold For Gain")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_SFG.png")
plt.clf()

sns.lmplot("MSP", "IV", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs % Increasing Value")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_IV.png")
plt.clf()

sns.lmplot("MSP", "DV", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs % Decreasing Value")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_DV.png")
plt.clf()

sns.lmplot("MSP", "TNV", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs Turnover")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_TNV.png")
plt.clf()

sns.lmplot("MSP", "BSI", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs Buyer Seller Index")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_BSI.png")
plt.clf()

sns.lmplot("MSP", "PTR", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs Price to Rent Ratio")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_PTR.png")
plt.clf()

sns.lmplot("MSP", "MHI", data, hue=None,fit_reg=False)
plt.title("Median Selling Price vs Market Health Index")
plt.tight_layout()
plt.savefig("../Figures/MA/MSP_vs_MHI.png")
plt.clf()

corrmat = data.corr()
sns.heatmap(corrmat, vmax=1., square=False)
plt.title("Correlation Matrix Plot")
plt.tight_layout()
plt.savefig("../Figures/MA/Correlation_matrix.png")
plt.clf()


#-----------------------------------------------------------------------
#Creating CSV with lag data for all features - to find optimal lag value (history)
#-----------------------------------------------------------------------
series = pd.Series.from_csv('../../Data/MA_dataset.csv',header=0)
df = pd.DataFrame()
for column in data:
    dataframe = pd.DataFrame()
    df = pd.DataFrame()
    if column != "MSP":
        for i in range(12,0,-1):
            df[str(column)+'t-'+str(i)]=data[column].shift(i)
            df[str(column)+'t-'+str(i)].replace('',np.nan, inplace=True)
        df.to_csv('../../Data/'+str(column)+'MA_Lagged_data.csv', index=False, sep=',')
    else:
        k =1
        for i in range(12,0,-1):
            df[str(column)+'t-'+str(i)]=data[column].shift(i)
        df[str(column)+'t'] = data[column].values
        df[str(column)+'t+'+str(k)] = data[column].shift(-1)
        df[str(column)+'t'].replace('',np.nan, inplace=True)
        df.to_csv('../../Data/'+str(column)+'MA_Lagged_data.csv', index=False, sep=',')
    
    # load data from new csv lag file
    dataframe = pd.read_csv('../../Data/'+str(column)+'MA_Lagged_data.csv', header=0)
    dataframe = dataframe.dropna()
    array = dataframe.values
    # split into input and output
    X = array[:,0:-1]
    y = array[:,-1]

    rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 1)
    fit = rfe.fit(X, y)
    # report selected features
    names = dataframe.columns.values[0:-1]
    # plot feature rank
    names = dataframe.columns.values[0:-1]
    ticks = [i for i in range(len(names))]
    plt.bar(ticks, fit.ranking_)
    plt.xticks(ticks, names, rotation=60)
    plt.title(str(column)+" Lag Rank")
    plt.xlabel("# Lags")
    plt.ylabel("Rank")
    plt.tight_layout()
    plt.savefig("../Figures/MA/"+str(column)+"Lag_data_rank.png")
    plt.clf()


#Optimal lag 8 - identified from manual inspection of plots created above 
#---------------------------------
#Final CSV for learning & forecast
#---------------------------------
df = pd.DataFrame()
for column in data:
    if column != "MSP":
        for i in range(8,0,-1):
            df[str(column)+'t-'+str(i)]=data[column].shift(i)
            df[str(column)+'t-'+str(i)].replace('',np.nan, inplace=True)
    else:
        k =1
        for i in range(8,0,-1):
            df[str(column)+'t-'+str(i)]=data[column].shift(i)
            if i == 1:
                df[str(column)+'t'] = data[column].values
                df[str(column)+'t+'+str(k)] = data[column].shift(-1)
                df[str(column)+'t'].replace('',np.nan, inplace=True)
                df[str(column)+'t+'+str(k)].replace('',np.nan, inplace=True)
            #print df.tail()
            df[str(column)+'t-'+str(i)].replace('',np.nan, inplace=True)
            
df = df.dropna()
df.to_csv('../../Data/MA_Lagged_data.csv', index=False, sep=',')


#--------------------------------------------------------
# Training and Testing Regression Models
#--------------------------------------------------------
# load data from new csv lag file
dataframe = pd.read_csv('../../Data/MA_Lagged_data.csv', header=0)
array = dataframe.values
# split into input and output
X_t = array[:,0:-2]
y_t = array[:,-2]
X_t1 = array[:,0:-1]
y_t1 = array[:,-2]

# fit random forest model                                
# Multi-output regressor
train_x = array[0:int(math.floor(array.shape[0]*0.9)),0:-2]
train_y = array[0:int(math.floor(array.shape[0]*0.9)),array.shape[1]-2:]
test_x = array[int(math.floor(array.shape[0]*0.9)):,0:-2]
test_y = array[int(math.floor(array.shape[0]*0.9)):,array.shape[1]-2:]

# PCA - For Dimensionality Reduction
# ----------------------------------
pca = PCA(n_components=10)

col = df.columns[:-2]

df1 = pd.DataFrame(train_x, columns=col)
data_scaled = pd.DataFrame(preprocessing.scale(df1),columns = df1.columns)
pca = PCA(n_components=20)
train_x_new = pca.fit_transform(train_x)
tempdf =  pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10', 'PC-11', 'PC-12', 'PC-13', 'PC-14', 'PC-15', 'PC-16', 'PC-17', 'PC-18', 'PC-19', 'PC-20'])

tempdf2 = tempdf.loc[:,:].abs()
index_list = tempdf2.index.values
feature_list = []

for i in index_list:
    max_pc = tempdf2.loc[i,:].max()
    for j in tempdf2.columns:
        if tempdf2.loc[i,j] == max_pc:
            feature_list.append(j)
feature_list = list(set(feature_list))
print feature_list

dataframe1 = dataframe[feature_list]
dataframe1.to_csv('../../Data/MA_New_Lagged_data.csv', index=False, sep=',')

array1 = dataframe1.values
train_x_pca = array1[0:int(math.floor(array1.shape[0]*0.9)),:]
train_y_pca = train_y
test_x_pca = array1[int(math.floor(array1.shape[0]*0.9)):,:]
test_y_pca = test_y

rmse_plain = {}
rmse_opt = {}

# Lasso Regression
#----------------------
print "----------------------------------------------"
print "LASSO"
print "----------------------------------------------"
#***********Plain**************#

reg = Lasso(random_state = 0)
y_pred = reg.fit(train_x, train_y).predict(test_x)
rmse_pl = np.sqrt(((y_pred - test_y) ** 2).mean()) #50.4044516747
print rmse_pl
rmse_plain['Lasso'] = rmse_pl

#***********Optimized**************#

pipe = Pipeline([('scl', StandardScaler()),
        ('reg', Lasso(random_state = 0))])

grid_param = {
    'reg__alpha': np.logspace(1, 4, num=25),
}

gs = (GridSearchCV(estimator=pipe, param_grid=grid_param, cv=3, n_jobs = -1))

gs = gs.fit(train_x_pca,train_y_pca)
best_param = gs.best_estimator_.steps[1][1].alpha #best_param = 10.0
print best_param
reg = Lasso(alpha = best_param, random_state = 0)
y_pred = reg.fit(train_x_pca, train_y_pca).predict(test_x_pca)

rmse_op = np.sqrt(((y_pred - test_y_pca) ** 2).mean()) #49.77626217
print rmse_op
rmse_opt['Lasso'] = rmse_op



# Gradient Boosting Regression
#-----------------------------
print "----------------------------------------------"
print "GRADIENT BOOSTING REGRESSOR"
print "----------------------------------------------"

#***********Plain**************#
reg = GradientBoostingRegressor(random_state = 0)
y_pred = MultiOutputRegressor(reg).fit(train_x, train_y).predict(test_x)
rmse_pl = np.sqrt(((y_pred - test_y) ** 2).mean()) #62.90984989
print rmse_pl
rmse_plain['GBR'] = rmse_pl

#***********Optimized**************#

pipe = Pipeline([('scl', StandardScaler()),
        ('reg', MultiOutputRegressor(reg))])

grid_param = {
    'reg__estimator__n_estimators': range(100, 2000, 100),
}

gs = (GridSearchCV(estimator=pipe, param_grid=grid_param, cv=3, n_jobs = -1))

gs = gs.fit(train_x_pca,train_y_pca)
best_param = gs.best_estimator_.steps[1][1].estimator.n_estimators #100
print best_param
reg = MultiOutputRegressor(GradientBoostingRegressor(n_estimators = best_param, random_state = 0))
y_pred = reg.fit(train_x_pca, train_y_pca).predict(test_x_pca)

rmse_op = np.sqrt(((y_pred - test_y_pca) ** 2).mean()) #57.87908645
print rmse_op
rmse_opt['GBR'] = rmse_op


# KNN Regression
#----------------------
print "----------------------------------------------"
print "KNN"
print "----------------------------------------------"
#***********Plain**************#
reg = KNeighborsRegressor()
y_pred = MultiOutputRegressor(reg).fit(train_x, train_y).predict(test_x)
rmse_pl = np.sqrt(((y_pred - test_y) ** 2).mean()) #
print rmse_pl
rmse_plain['KNN'] = rmse_pl

#***********Optimized**************#

pipe = Pipeline([('scl', StandardScaler()),
        ('reg', MultiOutputRegressor(reg))])

grid_param = {
    'reg__estimator__n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
}

gs = (GridSearchCV(estimator=pipe, param_grid=grid_param, cv=3, n_jobs = -1))

gs = gs.fit(train_x_pca,train_y_pca)
best_param = gs.best_estimator_.steps[1][1].estimator.n_neighbors #
print best_param

reg = MultiOutputRegressor(KNeighborsRegressor(n_neighbors = best_param))
pred_pre_t = reg.fit(train_x_pca, train_y_pca)
y_pred = pred_pre_t.predict(test_x_pca)

rmse_op = np.sqrt(((y_pred - test_y_pca) ** 2).mean()) #
print rmse_op
rmse_opt['KNN'] = rmse_op


# for plotting prediction and forecast graphs
#---------------------------------------------
temp = y_pred[-14 :,:]
actual_labels = temp[:,0]

#--------------------------------------------------------------------------
#For real forecast for 2 months (i.e 14 observations predicted recursively)
#--------------------------------------------------------------------------
print dataframe1.tail(1).values
randm = np.append(dataframe1.tail(1).values,test_y_pca[-1,:])
next_features = iter_feature(randm)
#print next_features
#print y_pred
next_y = test_y_pca[-1,:]

for i in np.arange(1,15,1):

    print "Iteration :", i
    if i == 1:
        arr_x = next_features
        arr_y = next_y
    else:
        print "skip"
    arr_x = np.array(arr_x).reshape((1, -1))
    pred_new_t = pred_pre_t.predict(arr_x)
    rmse_score_t_new = np.sqrt(((pred_new_t- arr_y) ** 2).mean())
    print "new rmse :", rmse_score_t_new
    forecast_rmse = np.append(forecast_rmse,rmse_score_t_new)
    arr_y = pred_new_t
    forecast_labels = np.append(forecast_labels,arr_y[0][0])
    val = np.append(arr_x,arr_y)
    arr_x = np.array(iter_feature(val))
print "forecast_labels :",forecast_labels
print "forecast_rmse :",forecast_rmse
print "actual_labels :",actual_labels

#-----------------------------------------
#Plotting RMSE Graphs based on predictions
#-----------------------------------------
#Graph between Jun - Jul days and predicted MSP
a = np.arange(len(actual_labels)) + 1
plt.plot(a, actual_labels, "m-")
plt.xticks(a, [str(i) for i in actual_days], rotation=45)
plt.xlabel("Predicted Days")
plt.ylabel("Median Selling Price")
plt.ylim(0, max(actual_labels))
plt.title("Timeline vs Median Selling Price")
plt.savefig("../Figures/MA/Predicted_MSP.png",bbox_inches='tight')
plt.tight_layout()
plt.clf()

#Graph between Aug - Sep days and Forecasted MSP
a = np.arange(len(forecast_labels)) + 1
plt.plot(a, forecast_labels, "m-")
plt.xticks(a, [str(i) for i in forecaste_days], rotation=45)
plt.xlabel("Forecasted Days")
plt.ylabel("Median Selling Price")
plt.ylim(0, max(forecast_labels))
plt.title("Timeline Forecast vs Median Selling Price")
plt.savefig("../Figures/MA/Forecasted_MSP.png",bbox_inches='tight')
plt.tight_layout()
plt.clf()

#Graph between Aug - Sep days and Forecasted MSP RMSE Score
a = np.arange(len(forecast_rmse)) + 1
plt.plot(a, forecast_rmse, "m-")
plt.xticks(a, [str(i) for i in forecaste_days], rotation=45)
plt.xlabel("Forecasted Days")
plt.ylabel("RMSE")
plt.ylim(0, max(forecast_rmse))
plt.title("Timeline Forecast vs RMSE")
plt.savefig("../Figures/MA/Forecaste_RMSE.png",bbox_inches='tight')
plt.tight_layout()
plt.clf()

