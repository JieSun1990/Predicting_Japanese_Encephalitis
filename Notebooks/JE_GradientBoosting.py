#!/usr/bin/env python
# coding: utf-8

# In[1]:


# need to install below to enable some features in plot_partial_dependence
get_ipython().run_line_magic('pip', 'install scikit-learn==0.24')


# ### Set up

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import time
import sys
from collections import Counter


# In[2]:


# ==================== Define functions ====================

def MyRsq(predict, actual): # Function to calculate R-squared
    dif = predict - actual
    dif = dif ** 2
    mse = np.mean(dif)
    rsq = 1 - mse / np.var(actual)
    return rsq

def Extract_Features(PD, Type = 'Full_Cov'): # Functions that only extract interesting features, remove unrelated columns in the pandas dataframe
    # Input PD is the entire pandas dataframe read from csv (result from Rds files of R)
    # Type is an approach that you want to run
    #   - Full_Cov: Keep all features --> remove x-y coordinates and FOI column
    #   - Only_Bio: Only keep Bioclimatic Features --> remove x-y coordinates, FOI column, and other not-bioclimatic columns
    # Output is an pandas dataframe containing only interesting features columns
    # Note: You need to check the names of columns that you want to remove
    
    if (Type == 'Full_Cov'):
        # Full Covariates --> Using all covariates --> remove x-y coordinates and FOI values columns (keep all features)
        if ('FOI' in PD.columns): # This check is for Endemic dataframe since it does not have FOI column    
            Interest_Features = PD.drop(['x', 'y', 'FOI'], axis = 1)
        else:
            Interest_Features = PD.drop(['x', 'y'], axis = 1)
    else:
        # Only Bio Covariates --> Only using bioclimatic --> remove x-y coordinates, FOI columns, and other columns that are not bioclimatic
        if ('FOI' in PD.columns): 
            Interest_Features = PD.drop(['x', 'y', 'DG_000_014bt_dens', 'Elv', 'Pigs', 'Pop_Count_WP_SEDAC_2015', 'Rice', 'UR', 'VD', 'FOI'], axis = 1)
        else:
            Interest_Features = PD.drop(['x', 'y', 'DG_000_014bt_dens', 'Elv', 'Pigs', 'Pop_Count_WP_SEDAC_2015', 'Rice', 'UR', 'VD'], axis = 1)
    return Interest_Features

np.random.seed(5) # set seed to regenerate the same random result




# In[3]:


# ==================== Set up parameters and Read data ====================

Typemodel = 'Full_Cov' # if Full Cov model --> More detail in above Extract_Features functions
#Typemodel = 'Only_Bio' # if only bio model 

Train_Portion = 0.7 # Portion of Train - Validate - Test
Validate_Portion = (1 - Train_Portion) / 2

resolution_grid = 400
Name_Grid_File = 'Grid_' + str(resolution_grid) + '_' + str(resolution_grid) + '.csv'

print('[Type Model] You have chosen ' + Typemodel)

CurDir = os.getcwd() + '/'
Data_All = CurDir + 'Generate/Python_CSV/EM_Imputed_Features_Study.csv' # 'Directory/to/your/EM_Imputed_Features_Study.csv'
Grid = CurDir + 'Generate/Grids_CSV/' + Name_Grid_File # 'Directory/to/your/Grid/' + Name_Grid_File
Data_EndemicDF = CurDir + 'Generate/Python_CSV/Imputed_Features_Endemic.csv' # 'Directory/to/your/Imputed_Features_Endemic.csv'

# Directory to the folder that you want this script exports files to (remember to have slash '/' at the end)
# Default is to create a subfolder named Python_Export and save result to that subfolder   
Savepath = CurDir + 'Generate/Python_Export/' 
if not os.path.exists(Savepath):
    os.makedirs(Savepath)

# Read csv and store in dataframe in pandas
AllData = pd.read_csv(Data_All)
Grid = pd.read_csv(Grid)  
EndemicDF = pd.read_csv(Data_EndemicDF)
EndemicDF = EndemicDF.drop(['FOI'], axis = 1) # remove FOI column (if it has)

# Remove Pop_2015 density (if wanted, since already have Pop_Count people) --> Check feature name again
# AllData = AllData.drop(['Pop_2015'], axis = 1)
# EndemicDF = EndemicDF.drop(['Pop_2015'], axis = 1)

# Check if matching coordinator
if (len(AllData.iloc[:, :2].merge(Grid.iloc[:, :2])) == len(AllData.iloc[:, :2])):
    print('[Checking Calibration] Matched Coordinators')
    # Recreate grid to match with AllData in case of nrow of 2 dataframe is different
    t = pd.merge(AllData.iloc[:, : 2].reset_index(), Grid.iloc[:, : 2].reset_index(), on=['x','y'], suffixes=['_1','_2'])
    Grid = Grid.iloc[t['index_2'], :]
    del t
else:
    sys.exit('[Stop Program] Grid and Data File do not match coordinators --> Check again')




# In[4]:


AllData.head()


# In[5]:


# ==================== Sampling Grids to define which Grids will be for Train-Validate-Test ====================

# Count freq of pix in each grid
Grid_column = Grid.iloc[:, 2]
Grid_column = np.array(Grid_column)
d = Counter(Grid_column)
grid_freq = np.array(list(d.values())) # number of pix in each grid_numb (belowed)
grid_numb = np.array(list(d.keys()))
del d

# ----- Preprocessing for Sampling train and validate -----
idx_grid_numb_less = np.where(grid_freq < 100)[0] # find idx of grid containing less than 100 pix --> these grids will be automaticly in training set
idx_grid_numb_high = np.where(grid_freq >= 100)[0] # find idx of grid containing more than 100 pix --> these grids will be randomly chosen for training

grid_numb_train_1 = grid_numb[idx_grid_numb_less]
grid_numb_sample = grid_numb[idx_grid_numb_high]

ngrid_train_2 = round(len(grid_numb_sample)*0.7) # 70% train --- 30% test (validate)
ngrid_validate = round(len(grid_numb_sample)*0.15)
ngrid_test = len(grid_numb_sample) - ngrid_train_2 - ngrid_validate
ngrid_train = len(grid_numb_train_1) + ngrid_train_2

print('[Sampling Grid] Training Grids: ' + str(ngrid_train) + ' ----- Validating Grids: ' + str(ngrid_validate) + ' ----- Testing Grids: ' + str(ngrid_test))

print('===== Sampling Model =====')    
grid_numb_sample_shuffle = np.copy(grid_numb_sample)
np.random.shuffle(grid_numb_sample_shuffle)
grid_numb_train_2 = grid_numb_sample_shuffle[:ngrid_train_2]
grid_numb_validate = grid_numb_sample_shuffle[ngrid_train_2:(ngrid_train_2 + ngrid_validate)]
grid_numb_test = grid_numb_sample_shuffle[(ngrid_train_2 + ngrid_validate):]
grid_numb_train = np.append(grid_numb_train_1, grid_numb_train_2)
del grid_numb_sample_shuffle, grid_numb_train_2

# ----- Take index of pixel for each sub-dataset
idx_train = np.where(np.in1d(Grid_column, grid_numb_train))[0]
idx_validate = np.where(np.in1d(Grid_column, grid_numb_validate))[0]
idx_test = np.where(np.in1d(Grid_column, grid_numb_test))[0]

Type = np.zeros(AllData.shape[0])
Type[idx_train] = 1 # index 1 for train
Type[idx_validate] = 2 # index 2 for validate
Type[idx_test] = 3 # index 3 for validate

# Saving Sampling Grid index having 3 columns: x-y coordinates, and Grid index
Coor_Grid_Index = AllData.iloc[:, 0:2]
Coor_Grid_Index = Coor_Grid_Index.assign(Type = pd.Series(Type).values)
filename_grid = 'Grid_Index_' + str(resolution_grid) + '.csv'
print('[Saving] Save Grid Index')
Coor_Grid_Index.to_csv(Savepath + filename_grid, sep='\t', encoding='utf-8')
print('[Saving] Done Saving Grid Index')




# In[6]:


# ==================== Create Train-Validate-Test dataset ====================
# ==================== SJ: Thinning the dataset ====================

# ===== Prepare Train =====
Train = AllData.iloc[idx_train, :]
row_na = Train.isnull().any(1) # check whether a row contains NA or not
#idx_row_na = row_na.nonzero()[0] # find index of row containing NA
#SJ: AttributeError: 'Series' object has no attribute 'nonzero'. This method is deprecated. See below
idx_row_na = row_na.to_numpy().nonzero()[0] # find index of row containing NA

print('[Preprocessing] Total Training containing NA: ' + str(len(idx_row_na)) + ' / ' + str(len(Train)) + ' ----- ' + 
      str(round(len(idx_row_na) / len(Train) * 100, 2)) + '%')
Train_Non_NA = Train.drop(Train.index[idx_row_na]) # remove row containing NA

# SJ: Thinning: take 10%
Train_Non_NA = Train_Non_NA.sample(frac = 0.1, replace = False, random_state = 1)

# ----- Extract Features for model -----
X_train = Extract_Features(Train_Non_NA, Typemodel)
Y_train = Train_Non_NA.FOI
Y_train = np.array(Y_train)

# ===== Prepare Validate =====
Validate = AllData.iloc[idx_validate, :]
row_na = Validate.isnull().any(1) # check whether a row contains NA or not
#idx_row_na = row_na.nonzero()[0] # find index of row containing NA
#SJ: AttributeError: 'Series' object has no attribute 'nonzero'. This method is deprecated. See below
idx_row_na = row_na.to_numpy().nonzero()[0] # find index of row containing NA

print('[Preprocessing] Total Validating containing NA: ' + str(len(idx_row_na)) + ' / ' + str(len(Validate)) + ' ----- ' + 
      str(round(len(idx_row_na) / len(Validate) * 100, 2)) + '%')
Validate_Non_NA = Validate.drop(Validate.index[idx_row_na]) # remove row containing NA

# SJ: Thinning: take 10%
Validate_Non_NA = Validate_Non_NA.sample(frac = 0.1, replace = False, random_state = 1)

# ----- Extract Features for model -----
X_validate = Extract_Features(Validate_Non_NA, Typemodel)
Y_validate = Validate_Non_NA.FOI
Y_validate = np.array(Y_validate)

# ===== Prepare Test =====
Test = AllData.iloc[idx_test, :]
row_na = Test.isnull().any(1) # check whether a row contains NA or not
#idx_row_na = row_na.nonzero()[0] # find index of row containing NA
#SJ: AttributeError: 'Series' object has no attribute 'nonzero'. This method is deprecated. See below
idx_row_na = row_na.to_numpy().nonzero()[0] # find index of row containing NA

print('[Preprocessing] Total Testing containing NA: ' + str(len(idx_row_na)) + ' / ' + str(len(Test)) + ' ----- ' + 
      str(round(len(idx_row_na) / len(Test) * 100, 2)) + '%')
Test_Non_NA = Validate.drop(Test.index[idx_row_na]) # remove row containing NA

# SJ: Thinning: take 10%
Test_Non_NA = Test_Non_NA.sample(frac = 0.1, replace = False, random_state = 1)

# ----- Extract Features for model -----
X_test = Extract_Features(Test_Non_NA, Typemodel)
Y_test = Test_Non_NA.FOI
Y_test = np.array(Y_test)

# ===== Prepare EndemicDF =====
row_na = EndemicDF.isnull().any(1) # check whether a row contains NA or not
#idx_row_na = row_na.nonzero()[0] # find index of row containing NA
#SJ: AttributeError: 'Series' object has no attribute 'nonzero'. This method is deprecated. See below
idx_row_na = row_na.to_numpy().nonzero()[0] # find index of row containing NA

print('[Preprocessing] Total EndemicDF containing NA: ' + str(len(idx_row_na)) + ' / ' + str(len(EndemicDF)) + ' ----- ' + 
      str(round(len(idx_row_na) / len(EndemicDF) * 100, 2)) + '%')
EndemicDF_Non_NA = EndemicDF.drop(EndemicDF.index[idx_row_na]) # remove row containing NA
# ----- Extract Features for model -----
X_endemic = Extract_Features(EndemicDF_Non_NA, Typemodel)

# ----- Find max_ft for randomforest regression = numft / 3 (default) -----
num_ft = np.floor(X_train.shape[1]/3)
if (num_ft != X_train.shape[1]/3):
    num_ft = num_ft + 1
num_ft = int(num_ft)


# ## Gradient Boosting

# In[7]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence


# #### Model building

# In[8]:


# ==============Hyperparameter Tuning with GBoosting

# Random Search
# tune colsample_bytree,n_estimators,max_depth, lambda
# There is a relationship between the number of trees (n_estimators) and the depth of each tree (max_depth)
# Create the parameter grid: gbm_param_grid
# didn't tune alpha as I tuned lambda

gbr_param_grid = {
    'learning_rate': [0.0001,0.001, 0.01, 0.05,0.1,0.15,0.3],
    'n_estimators': range(50,250,10),
    'max_depth': [4,6,8,10],
    'min_samples_leaf': [5,10,15], 
    'subsample': np.arange(3,8)/10,
    'max_features':['auto','sqrt','log2']
}
# Instantiate the regressor: gbm
gbr = GradientBoostingRegressor()
# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(param_distributions=gbr_param_grid, 
                                    estimator=gbr, 
                                    scoring="neg_mean_squared_error",  
                                    cv=4, #kfolds
                                    verbose=1,
                                   n_jobs = -1)
# Fit randomized_mse to the data
randomized_mse.fit(X_train, Y_train) #actually, should just use X_train as the combined train and test, as CV will have its own split

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest training RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
print("Lowest training MSE found: ", np.abs(randomized_mse.best_score_))


# In[9]:


bestp = randomized_mse.best_params_
bestp


# In[8]:


# #new version
bestp = {'subsample': 0.7,
 'n_estimators': 220,
 'min_samples_leaf': 10,
 'max_features': 'sqrt',
 'max_depth': 10,
 'learning_rate': 0.1}


# Use the best params to:
# 
# - redo the fitting process, 
# - predict on test, 
# - calculate test MSE, 
# - get CI,
# - plot the feature importance

# In[9]:


# ============== Refit GBoost with the optimal params

gbr = GradientBoostingRegressor(n_estimators = bestp['n_estimators'],
                          max_depth = bestp['max_depth'],
                                learning_rate = bestp['learning_rate'],
                                subsample = bestp['subsample'],
                                min_samples_leaf = bestp['min_samples_leaf'],
                                random_state = 123)
gbr.fit(X_train, Y_train)
yhat = gbr.predict(X_validate)


# In[12]:


# ============== Save the model
filename_model = Savepath + 'ModelGB_' + '.model'
print('[Saving] Save training model')
pickle.dump(gbr, open(filename_model, 'wb'))


# In[13]:


# You can load the model by the following way

# filename_model = Savepath + 'ModelGB_' + '.model'
# gbr = pickle.load(open(filename_model, 'rb'))


# #### Evaluate model performance

# In[14]:


# ============== 1. MSE
rmse = np.sqrt(mean_squared_error(Y_validate, yhat))
mse = mean_squared_error(Y_validate, yhat)
print("Validate RMSE: %f" % (rmse))
print("Validate MSE: %f" % (mse))


# The MSE is relatively low compared to RF. However, it is slightly higher than XGBoost. 

# In[10]:


# ============== 2. Deviance (train/test error)

test_score = np.zeros((bestp['n_estimators'],), dtype=np.float64)
for i, yhat in enumerate(gbr.staged_predict(X_validate)):
    test_score[i] = gbr.loss_(yhat,Y_validate)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(bestp['n_estimators']) + 1, gbr.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(bestp['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()


# The i-th train_score is the deviance (= loss) of the model at iteration i on the in-bag sample. Calculation of deviance is dependent on the loss function chosen. In this case, the default is 'ls', i.e. least-square error. 
# 
# Testing set deviance decreases as iterations increases, but tends to remain stable after 25 iterations, suggesting that the model is preventing overfitting. This is mainly due to the shrinkage parameter. 

# #### Model output

# In[11]:


# ============== 1. get feature importances
# it is the same API interface like for ‘scikit-learn’ models

# feature importance
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X_validate.columns)[sorted_idx])
plt.title('Feature Importance - GBR')

# permutation importance 
result = permutation_importance(gbr, X_validate, Y_validate, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(X_validate.columns)[sorted_idx])
plt.title("Permutation Importance (validation set)")

figname2 = Savepath + 'GBR_permu_feature_imp.png'
plt.savefig(figname2)

fig.tight_layout()
plt.show()


# In[13]:


# list of top 10 features based on permutation importance

top10 = np.flip(np.array(X_train.columns)[sorted_idx])[:10]
top10


# MDI(mean decrease in impurity)-based feature importance and permutation importance tend to give slightly different results. This is due to the different measurement philosophy. 
# 
# The **permutation importance** is often more robust against the cardinality of variables. Based on the scikit-learn literature: impurity-based feature importance for trees are strongly biased and favor high cardinality features (typically numerical features) over low cardinality features such as binary features or categorical variables with a small number of possible categories (https://scikit-learn.org/stable/modules/permutation_importance.html).
# 
# In this example, both importance scores are more or less consistent with each other. 
# 
# Note, that in our previous analysis using PCA, we have identified a number of features that are closely related. When two features are correlated and one of the features is permuted, the model will still have access to the feature through its correlated feature. This will result in a lower importance value for both features, where they might actually be important(https://scikit-learn.org/stable/modules/permutation_importance.html). Therefore we may want to cluster **correlated** features BEFORE running the gradient boosting and calculating permutation importance.

# In[25]:


# ============== 2. get dependence plots
# Partial dependence plots
_, ax = plt.subplots(figsize=(10, 10))
ax.set_title( 'Partial dependence of top 10 features on FOI\nGradient Boosting')
display2 = plot_partial_dependence(gbr,X_validate, top10[:5],ax = ax)


# In[14]:


# Both PD and ICE (individual conditional expectation) plots
# limit to 100 ICE curves to avoid overcrowding
for i in range(5):
    display3 = plot_partial_dependence(gbr,X_validate, [top10[i]], kind = 'both', subsample = 100)
    display3.figure_.suptitle(
        'Partial dependence with individual conditional expectation\n of top features on FOI - Gradient Boosting'
    )
    display3.figure_.subplots_adjust(wspace=0.2, hspace=1)


# Here we look at the partial dependency plots and the individual conditional expectation plots. 
# 
# #### PDP
# The above one-way PDP specifies the relationship between target variable (FOI) and the top 10 input feature, based on the permutation importance score. The line indicates the **average** effect of the input feature.
# The y axis of PDP represents the **marginal** impact of the independent variable to the dependent variable. 
# 
# - We can see that the marginal effects of each variable on the FOI are non-linear. For Bio_11 and Bio_18, they appear to be stepwise. 
# 
# Note that a major assumption of the PDP is that the features are **independent**. When there are interactions between feature, PDP tend to obscure the heterogeneous relationship created by interactions. Hence we check ICE for more insights.
# 
# 
# #### ICE
# An ICE plot visualizes the dependence of the prediction on a feature for each sample separately with **one line per sample**. Each line represents a sample, describing the effect of that feature on FOI, given all other features remain constant. An ICE plot can highlight the variation in the fitted values across the range of a feature. This suggests where and to what extent heterogeneities might exist.
# 
# The y axis of ICE represents the **expected** value of the dependent variable. 
# 
# - Comparing the ICE plots with PDP, individual contributions of each X variable vary by sample. For example, Bio_18 values in certain ranges increase the FOI sharply, while the average effect remains rather constant.
# 
# Note that with highly correlated features, ICE plots may produce invalid data points as well. 
# 
# #### More references
# 
# - Goldstein, Alex, Adam Kapelner, Justin Bleich, and Emil Pitkin. 2013. “Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation.” Journal of Computational and Graphical Statistics 24 (September). doi:10.1080/10618600.2014.907095.
# 
# - Hastie, T., R. Tibshirani, and J. Friedman. 2013. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Series in Statistics. Springer New York. https://books.google.de/books?id=yPfZBwAAQBAJ.
# 
# - Molnar, Christoph. 2019. Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.
# 
# 

# #### Model prediction

# In[29]:


# ============== Prediction intervals
# This is again using quantile regression 

# confidence level
alpha = 0.95
# quantile regression 
clf = GradientBoostingRegressor(loss = 'quantile',alpha = alpha,
    n_estimators = bestp['n_estimators'],
                          max_depth = bestp['max_depth'],
                                learning_rate = bestp['learning_rate'],
                                subsample = bestp['subsample'],
                                min_samples_leaf = bestp['min_samples_leaf'],
                                random_state = 123)
# upper
clf.fit(X_train, Y_train)
y_upper_gbm = clf.predict(X_validate)

# lower
clf.set_params(alpha = 1.0 - alpha)
clf.fit(X_train, Y_train)
y_lower_gbm = clf.predict(X_validate)

# yhat
clf.set_params(loss='ls')
clf.fit(X_train, Y_train)
y_pred_gbm = clf.predict(X_validate)


# In[30]:


# dataframe
dat = np.concatenate((Y_validate.reshape(-1,1), 
                      y_pred_gbm.reshape(-1,1), 
                      y_lower_gbm.reshape(-1,1), 
                      y_upper_gbm.reshape(-1,1)), axis = 1)
dat = pd.DataFrame(dat, columns = ['y','yhat','lower','upper'])
dat.head()


# In[31]:


# sort the dataframe for plotting
dat2 = dat.sort_values('y')
dat2.head()


# In[32]:


# plot1
fig, ax = plt.subplots(figsize = (15,10))
plt.scatter(dat2['y'], dat2['y'],label = 'original validate y')
plt.scatter(dat2['y'], dat2['yhat'],label = 'predicted y')
ax.plot(dat2['y'], dat2['upper'], '--', color = 'b')
ax.plot(dat2['y'], dat2['lower'], '--', color = 'b')
ax.fill_between(dat2['y'], dat2['lower'],dat2['upper'], color = 'blue',alpha = .1)
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('95% prediction interval of Y versus the original Y')
plt.legend()

# plot2
fig, ax = plt.subplots(figsize = (15,10))
plt.scatter(dat2['y'], dat2['y'],label = 'original validate y')
plt.scatter(dat2['y'], dat2['yhat'],label = 'predicted y')
# ax.plot(dat2['y'], dat2['upper'], '--', color = 'b')
# ax.plot(dat2['y'], dat2['lower'], '--', color = 'b')
# ax.fill_between(dat2['y'], dat2['lower'],dat2['upper'], color = 'blue',alpha = .1)
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Predicted Y versus the Original Y')
plt.legend()


# The above plots show that: 
# 
# - The quantile regression is not really capturing the true Y as Y goes beyond 0.25.
# 
# - The quantile regression is capturing the true Y for Y < 0.25. 
# 
# - GBM tends to under-estimat the actual Y if Y > 0.25. 
# 
# What characteristics do the points with Y > 0.25 have? 
# 
# 1. Check distribution of Y_validate
# 2. Check the X_validate attributes for Y_validate > .25

# In[33]:


print(f'The proportion of Y_validate > 0.25 out of the whole validation sample is: ',sum(Y_validate > .25)/len(Y_validate))


# In[34]:


# check distribution of Y_validate
n, bins, patches = plt.hist(x=Y_validate, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value - Y_validate')
plt.ylabel('Frequency')
plt.title('Distribution of Y_validate')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# In[35]:


# summary of the top 10 features in X_validate
X_validate[top10].describe()


# In[36]:


# summary of the top 10 features in X_validate, for Y_validate > 0.25
Validate_Non_NA[Validate_Non_NA.FOI > 0.25][top10].describe()


# The above suggested that: 
# 
# - Y_validate only takes a very small proportion (7%) of the total validation dataset. Therefore failing to cover the true Y in the 95% prediction interval is not impacting the overall model performance much. 
# - The under-estimation of FOI in the validation set for this range is not captured by the usual model evaluation methods, such as test MSE, plot of train-test error, etc.  
# - By reviewing the attributes whose target FOI > 0.25, we see that:
# 
# 1). Bio_11 is much larger than the whole validation set
# 
# 2). Bio_04 is much smaller than the whole validation set
# 
# 3). Other top 10 features can be compared in the similar manner.
# 
# - For inference, we should be careful using GBM for records whose X variables fall within the range above. 

# In[ ]:





# In[ ]:




