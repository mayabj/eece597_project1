##############################################################################################################
# This script trains GradientBoosting regression models for given hyperparameter values and saves the model
##############################################################################################################

# Import packages

import time
import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# Declare Variables
header=["Max depth", "No of Trees", "MAE", "R2 Score"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
k_fold_cross_validation = int(5) ## 5 fold cross validation
#max_tree_depth=int(6)
max_tree_depth_list = ['3']
no_of_trees_list = ['100','200','300','400']

# Specify File Paths
train_data_file_path_reg = 'G:/ProjectData/routing_data_train/mvto-1000-1_train_X.csv'
train_label_file_path_reg = 'G:/ProjectData/routing_data_train/mvto-1000-1_train_y.csv'

# ===================== Functions =================================================================

def load_train_data_reg(train_data_file_path,train_label_file_path):
    """
    Reads in Train Data
    @param train_data_file_path:
    @param train_label_file_path:
    @return : data_train_X,data_train_y
    """
    data_train_X = np.genfromtxt(train_data_file_path, delimiter=',', dtype=float)
    data_train_y = np.genfromtxt(train_label_file_path, delimiter=',', dtype=int, usecols=0)

    # Find how many routing runs are captured in the Y_labels.csv
    # Number of routing runs = number of 1s
    number_of_runs_train = list(data_train_y.flatten()).count(1)
    return data_train_X,data_train_y




# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('Max Depth values :', max_tree_depth_list)
print('No: of trees (n_estimators) : ', no_of_trees_list)
print('--------------------------------------------------------------------------------------')
print('\nLoad Data : Train')
(data_train_X_reg, data_train_y_reg) = load_train_data_reg(train_data_file_path_reg, train_label_file_path_reg)
columns_X = data_train_X_reg.shape[1]  # no: of features


for m in max_tree_depth_list:
    max_tree_depth = int(m)
    for n in no_of_trees_list:
        no_of_trees = int(n)
        print('\nNo: of trees (n_estimators) : ', no_of_trees)
        print('Start Time : ', time.strftime("%H:%M:%S", time.localtime()))

        print('  Load Regression Model')
        gb = GradientBoostingRegressor(random_state=0, max_depth=max_tree_depth, n_estimators=no_of_trees)
 
        print('  Fit model to the data')
        clf = gb.fit(data_train_X_reg, data_train_y_reg)

        #Calculate Metrics
        print('  Calculate metrics for Train data')
        predicted_train_reg = gb.predict(data_train_X_reg)
        mae = mean_absolute_error(data_train_y_reg, predicted_train_reg)
        r2_score_val = r2_score(data_train_y_reg, predicted_train_reg)
        print(header)
        print(str(max_tree_depth)+','+ str(no_of_trees) +','+ str(mae) + ',' + str(r2_score_val))
        print('    Save the model : '+'mvto-1000-1_gradb_reg_' + str(max_tree_depth) +'d' + str(no_of_trees) + 't')
        save_file_name = 'mvto-1000-1_gradb_reg_' + str(max_tree_depth) +'d' + str(no_of_trees) + 't'
        joblib.dump(clf, save_file_name)
        print('\nEnd Time : ', time.strftime("%H:%M:%S", time.localtime()))

print('\n--------------------------------------------------------------------------------------')


