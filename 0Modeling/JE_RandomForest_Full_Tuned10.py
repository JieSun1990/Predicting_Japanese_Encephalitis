#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
from scipy.special import expit, logit

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import validation_curve


# In[3]:


# 100% 


# ==================== Define functions ====================

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
Savepath = CurDir + 'Generate/Python_Export/' #@@
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


# ==================== Create Train-Validate-Test dataset ====================

# ===== Prepare Train =====
Train = AllData.iloc[idx_train, :]
row_na = Train.isnull().any(1) # check whether a row contains NA or not
#idx_row_na = row_na.nonzero()[0] # find index of row containing NA
#SJ: AttributeError: 'Series' object has no attribute 'nonzero'. This method is deprecated. See below
idx_row_na = row_na.to_numpy().nonzero()[0] # find index of row containing NA

print('[Preprocessing] Total Training containing NA: ' + str(len(idx_row_na)) + ' / ' + str(len(Train)) + ' ----- ' + 
      str(round(len(idx_row_na) / len(Train) * 100, 2)) + '%')
Train_Non_NA = Train.drop(Train.index[idx_row_na]) # remove row containing NA

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


# In[6]:


m_name = 'RF'
print('===== Training Random Forest =====')

# # ============== Hyperparameter Tuning with RF

# # most important params to tune: n_estimators, max_features

# # Number of trees in random forest
# n_estimators = [100,200,400,800,1200]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [3,8,15,20]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1,4,8]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_leaf': min_samples_leaf}

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, 
#                                param_distributions = random_grid, 
#                                cv = 4, verbose=1, scoring="neg_mean_squared_error", 
#                                random_state=42, n_jobs = -1)

# # Fit the random search model
# rf_random.fit(X_train, Y_train)

# # Print the best parameters and lowest MSE
# bestp = rf_random.best_params_
# print("Best parameters found: ", bestp)

# copy the bestp from 10% pilot and observe the MSE plot
bestp = {'n_estimators': 180, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 40}

# save best parameters to csv
bestp_pd = pd.DataFrame(list(bestp.items()), columns = ['Parameters','Value'])
filename_bestp = 'Bestp_' + m_name + '_' + Typemodel + '_' + str(resolution_grid) + '.csv'
bestp_pd.to_csv(Savepath + filename_bestp, sep='\t', encoding='utf-8')

# ============== Refit RF with the optimal params

regr = RandomForestRegressor(n_estimators = bestp['n_estimators'], 
                             min_samples_leaf= bestp['min_samples_leaf'], 
                             max_features= bestp['max_features'], 
                             max_depth= bestp['max_depth'],random_state = 123, n_jobs = -1)

start_time = time.time()
regr.fit(X_train, Y_train)
end_time = time.time()
print('[Training] Finish training')
training_time = round(end_time - start_time, 5) # seconds
print('Training Time: ' + str(training_time) + ' seconds')

yhat = regr.predict(X_test)

# ============== Save the model, note that the model is VERY big

# filename_model = Savepath + 'ModelRF_' + '.model'
# print('[Saving] Save training model')
# pickle.dump(regr, open(filename_model, 'wb'))

# You can load the model by the following way

# filename_model = Savepath + 'ModelRF_' + '.model'
# regr = pickle.load(open(filename_model, 'rb'))


# In[9]:


# ============== 1. MSE
mse_train = mean_squared_error(Y_train, regr.predict(X_train))
mse_test = mean_squared_error(Y_test, yhat)
print("train MSE: %f" % (mse_train))
print("test MSE: %f" % (mse_test))

# Export csv of MSE
print('[Saving] Save MSE evaluation')        
Result_pd = pd.DataFrame(data = {'mse_train':[mse_train], 'mse_test':[mse_test], 'time_train':[training_time]})   
filename_mse = 'MSE_' + m_name + '_' + Typemodel + '_' + str(resolution_grid) + '.csv'
Result_pd.to_csv(Savepath + filename_mse, sep='\t', encoding='utf-8')

# ============== 2. MSE on train/validation
rf = RandomForestRegressor(n_estimators = bestp['n_estimators'], 
                             min_samples_leaf= bestp['min_samples_leaf'], 
                             max_features= bestp['max_features'], 
                             max_depth= bestp['max_depth'],random_state = 123)
mse_train = []
mse_validate = []
list_nb_trees = np.arange(10,bestp['n_estimators'],20)

for tree in list_nb_trees:
    rf.n_estimators = tree
    rf.fit(X_train, Y_train)
    y_train_predicted = rf.predict(X_train)
    y_validate_predicted = rf.predict(X_validate)
    mse_train.append(mean_squared_error(Y_train, y_train_predicted))
    mse_validate.append(mean_squared_error(Y_validate, y_validate_predicted))
    #print("Iteration: {} Train mse: {} Test mse: {}".format(iter, mse_train, mse_test))
    

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Random Forest')
plt.plot(list_nb_trees, mse_train, 'b-',
         label='Training Set MSE')
plt.plot(list_nb_trees, mse_validate, 'r-',
         label='Validation Set MSE')
plt.legend(loc='upper right')
plt.xlabel('No. trees')
plt.ylabel('MSE')
plt.rc('font',size = 15)
plt.rc('axes',labelsize = 15)
fig.tight_layout()
plt.show()

# Save MSE plot 
print('[Saving] Save MSE plot')
filename_mse_plot = 'MSE_' + m_name + '_'+ Typemodel + '_' + str(resolution_grid) + '.png'
plt.savefig(Savepath + filename_mse_plot)


# ### Model output

# In[23]:


# ============== 1. get feature importances

# feature importance
feature_importance = regr.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(6, 10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X_test.columns)[sorted_idx])
plt.title('Random Forest')
plt.xlabel('Feature Importance Scores')
plt.rc('font',size = 15)
plt.rc('axes',labelsize = 15)
plt.tight_layout()
# Save feature importance and plot 
print('[Saving] Save variable importance ranking plot')
filename_varimp_plot = 'Varimp' + m_name + '_' + Typemodel + '_' + str(resolution_grid) + '.png'
plt.savefig(Savepath + filename_varimp_plot)

print('[Saving] Save variable importance ranking')
data = {'Name': X_endemic.columns, 'Importance': feature_importance}
importance_df = pd.DataFrame(data)
importance_df["Std"] = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
importance_df = importance_df.sort_values(by = 'Importance', ascending = False)
filename_varimp = 'Varimp' + m_name + '_' + Typemodel + '_' + str(resolution_grid) + '.csv'
importance_df.to_csv(Savepath + filename_varimp, sep='\t', encoding='utf-8')

# list of top 10 features based on permutation importance
top10 = np.flip(np.array(X_train.columns)[sorted_idx])[:10]
print('Top 10 features are: ', top10)

# Export top10
print('[Saving] Save top10 features')        
Result_pd = pd.DataFrame(data = {'top10':top10})      
filename_mse = 'top10_' + m_name + '_' + Typemodel + '_' + str(resolution_grid) + '.csv'
Result_pd.to_csv(Savepath + filename_mse, sep='\t', encoding='utf-8')


# In[11]:


# ============== 2. get dependence plots
# Partial dependence plots
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title( 'Random Forest')
display2 = plot_partial_dependence(regr,X_test, top10[:3],ax = ax)
display2.axes_[0][0].set_ylabel('Predicted Logit Test FOI')

# Save partial dependence plots
print('[Saving] Save partial dependence plots')
filename_pd_plot = 'PDP' + m_name + '_' + Typemodel + '_' + str(resolution_grid) + '.png'
ax.figure.savefig(Savepath + filename_pd_plot)

# ICE plots
fig, ax = plt.subplots(figsize=(10, 5))
display3 = plot_partial_dependence(regr,X_test, top10[:3],ax = ax, kind = 'both', subsample = 100)
ax.set_title('Random Forest')
display3.axes_[0][0].set_ylabel('Predicted Logit Test FOI')

# Save ICE plots 
print('[Saving] Save individual conditional expectation plots')
filename_ice_plot = 'ICE_RF_' + Typemodel + '_' + str(resolution_grid) + '.png'
ax.figure.savefig(Savepath + filename_ice_plot)


# In[15]:


# ============== 2. Quantile regression 
def pred_ints(model, X, percentile = 95):
    print('[Predicting CI] Start predicting CI of FOI')
    start_time = time.time()
    err_down = []
    err_up = []
    for x in range(len(X)): #for each row of data
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X.iloc[x,:].values.reshape(1,-1))[0])
        err_down.append(np.percentile(preds, (100-percentile)/2.))
        err_up.append(np.percentile(preds, 100-(100-percentile)/2.))
    end_time = time.time()
    print('[Predicting CI] Finish predicting CI of FOI')
    training_time = round(end_time - start_time, 5) # seconds
    print('Predicting Time: ' + str(training_time) + ' seconds')
    
    return err_down, err_up

err_down, err_up = pred_ints(regr, X_test, percentile = 95)
ylow = np.array(err_down)
yup = np.array(err_up)

# ----- Export csv of TestDF (Coor and result)
print('[Saving] Save predicted test FOI with coords')
Coor = Test_Non_NA.iloc[:, 0: 2]
Coor = Coor.assign(Actual = pd.Series(Y_test).values,
                   Predict = pd.Series(yhat).values,
                   Lower = pd.Series(ylow).values,
                   Upper = pd.Series(yup).values)
filename_test = 'Test_FOI_'+ m_name +'_quantile_' + Typemodel + '_' + str(resolution_grid) + '.csv'
Coor.to_csv(Savepath + filename_test, sep='\t', encoding='utf-8')


# In[18]:


def plotci_2(Y_test, yhat, ylow, yup, m_name):
    # dataframe
    dat = np.concatenate((Y_test.reshape(-1,1), 
                          yhat.reshape(-1,1), 
                          ylow.reshape(-1,1), 
                          yup.reshape(-1,1)),axis = 1)
    dat = pd.DataFrame(dat, columns = ['y','yhat','lower','upper'])

    # sort the dataframe for plotting
    dat2 = dat.sort_values('y')
    
    # take 1% and replot the above CI
    dat2 = dat2.sample(frac = 0.01, replace = False, random_state = 123)

    fig, ax = plt.subplots(figsize = (10,10))
    plt.scatter(dat2['y'], dat2['yhat'],label = 'Predicted test FOI')
    plt.plot([0.05,0.45],[0.05,0.45], 'k--')
    plt.errorbar(dat2['y'], dat2['yhat'], np.array(dat2['lower'], dat2['upper']), fmt='o',alpha = 0.5)

    plt.xlabel('Actual FOI')
    plt.ylabel('Predicted FOI')
    plt.ylim(0,0.5)
    plot_name = m_name
    plt.title(plot_name)
    plt.legend()
    plt.rc('font',size = 15)
    plt.rc('axes',labelsize = 15)
    plt.tight_layout()

    # Save thinned CI plot 
    print('[Saving] Save thinned CI plot')
    filename_ci_plot = 'CI_'+ m_name +'_' + Typemodel + '_' + str(resolution_grid) + '.png'
    plt.savefig(Savepath + filename_ci_plot)
    
plotci_2(Y_test, yhat, ylow, yup, 'RF')


# In[19]:


# ============== 2. Quantile regression
# predict and convert
err_down, err_up = pred_ints(regr, X_endemic, percentile = 95)
Y_Predict_Endemic = expit(regr.predict(X_endemic))
ylow_endemic = expit(np.array(err_down))
yup_endemic = expit(np.array(err_up))

# ----- Export csv of EndemicDF (Coor and result)
print('[Saving] Save predicted FOI with coords')
Coor = EndemicDF_Non_NA.iloc[:, 0: 2]
Coor = Coor.assign(Predict = pd.Series(Y_Predict_Endemic).values,
                  Lower = ylow_endemic,
                  Upper = yup_endemic)
filename_endemic = 'Endemic_FOI_RF_Quantile Regression_' + Typemodel + '_' + str(resolution_grid) + '.csv'
Coor.to_csv(Savepath + filename_endemic, sep='\t', encoding='utf-8')

