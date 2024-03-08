###############################################################################################
# This script is used to find optimum Hyperparameters for GradientBoosting regression models
# Hyperparameters considered are :
#    (1) number of trees (n_estimators) in the ensemble
# max_depth is set to the default value of 3
#
# no:of trees (n_estimators) is varied in a grid search approach : eg:5,50,100,200..1000 (these are passed to the script from another script)
# 5-fold cross validation is performed for hyperparameter tuning and following metrics are calculated for both Train & Test data
# ==> Mean Absolute Error (MAE) , R2_score
# For a given no: of trees, the cumulative contribution across all folds is calculated using weighted average
# The n_estimators value corresponding to highest R2_score is chosen as the optimum one

#######################################################################################################

# Import packages

import argparse
import time
import csv
import numpy as np
import os
import re
import pprint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# Declare Variables
header=["No of Trees", "Fold", "No of Samples in Train", "No of Samples in Test fold", "MAE (Train)", "MAE (Test)", "R2_score (Train)", "R2_score (Test)"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
k_fold_cross_validation = int(5) ## 5 fold cross validation
#max_tree_depth=int(6)
max_tree_depth = int(3)

# Specify File Paths
train_data_label_path_reg = 'G:/ProjectData/mvto_50-1000_reg_data_for_cv/'


# ===================== Functions =================================================================

def capture_ckt_info(train_data_label_path_reg, train_data_file_reg, train_label_file_reg):
    """
    Capture circuit information
    @param train_data_label_path_reg:
    @param train_data_file_reg:
    @param train_label_file_reg:
    @return : list_ckt_names_reg
    """

    # regex_data = re.compile('.*(%s).*'%train_data_file_reg)
    # regex_label = re.compile('.*(%s).*'%train_label_file_reg)
    regex_data = re.compile(".*({}).*".format(train_data_file_reg))
    regex_label = re.compile(".*({}).*".format(train_label_file_reg))
    list_data_files_reg = []
    list_label_files_reg = []
    list_ckt_names_reg = []
    # Capture circuit information from binary regification data
    for root, dirs, files in os.walk(train_data_label_path_reg):
        for file in files:
            if regex_data.match(file):
                list_data_files_reg.append(file)
                list_ckt_names_reg.append(file.split(train_data_file_reg)[0])
            if regex_label.match(file):
                list_label_files_reg.append(file)

    no_of_ckts_reg = len(list_ckt_names_reg)
    print('             No of Circuits in data set : ', no_of_ckts_reg)
    print('Circuit Names :')
    pp = pprint.PrettyPrinter(width=100, compact=True)
    pp.pprint(list_ckt_names_reg)
    return list_ckt_names_reg


def organize_train_test_reg(list_ckt_names_reg,test_set,train_data_label_path_reg):
    """
    Organize data
    @param list_ckt_names_reg:
    @param test_set:
    @param train_data_label_path_reg
    @return : data_train_X,data_train_y, data_test_X, data_test_y
    """
    train_set = list(set(list_ckt_names_reg).difference(test_set))
    #print('         Test set : ', test_set)
    #print('         Train set : ')
    pp = pprint.PrettyPrinter(width=100, compact=True)
    #pp.pprint(train_set)
    #print('created train set : ', time.strftime("%H:%M:%S", time.localtime()))
    test_file_root_list = np.char.add(train_data_label_path_reg, test_set)
    test_data_file_list = np.char.add(test_file_root_list, train_data_file_reg)
    test_label_file_list = np.char.add(test_file_root_list, train_label_file_reg)

    train_file_root_list = np.char.add(train_data_label_path_reg, train_set)
    train_data_file_list = np.char.add(train_file_root_list, train_data_file_reg)
    train_label_file_list = np.char.add(train_file_root_list, train_label_file_reg)

    # Concatenate info in csv files
    columns_X = np.genfromtxt(train_data_file_list[0], delimiter=',', dtype=float).shape[1]
    #print('columns :', columns_X)
    data_train_X = np.empty((0, columns_X))
    data_train_y = np.empty(0)
    data_test_X = np.empty((0, columns_X))
    data_test_y = np.empty(0)

    for file in train_data_file_list:
        data_train_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_train_X = np.concatenate((data_train_X, data_train_X_temp), axis=0)

    for file in train_label_file_list:
        data_train_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_train_y = np.concatenate((data_train_y, data_train_y_temp))

    for file in test_data_file_list:
        data_test_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_test_X = np.concatenate((data_test_X, data_test_X_temp), axis=0)

    for file in test_label_file_list:
        data_test_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_test_y = np.concatenate((data_test_y, data_test_y_temp))

    return data_train_X, data_train_y, data_test_X, data_test_y


def perform_k_fold_cross_validation(list_ckt_names_reg, k_fold_cross_validation, test_ckts_array, writer, gb):
    """
    Perform K-fold cross validation
    @param list_ckt_names_reg:
    @param k_fold_cross_validation:
    @param test_ckts_array:
    @param writer:
    @param gb:
    """
    train_samples_for_each_fold_list = []
    test_samples_for_each_fold_list = []
    mae_weighted_train_list = []
    mae_weighted_test_list = []
    r2score_weighted_train_list = []
    r2score_weighted_test_list = []
    for i in range(0, k_fold_cross_validation):
        print('           Fold : ',i)
        test_set = test_ckts_array[i]
        # print('test set:', test_set)
        data_train_X_reg, data_train_y_reg, data_test_X_reg, data_test_y_reg = organize_train_test_reg(list_ckt_names_reg, test_set, train_data_label_path_reg)
        # Fit model to data
        clf = gb.fit(data_train_X_reg, data_train_y_reg)

        test_samples = data_test_y_reg.shape[0]
        test_samples_for_each_fold_list.append(test_samples)
        train_samples = data_train_y_reg.shape[0]
        train_samples_for_each_fold_list.append(train_samples)

        predicted_train_reg = gb.predict(data_train_X_reg)
        predicted_test_reg = gb.predict(data_test_X_reg)
        mae_train = mean_absolute_error(data_train_y_reg, predicted_train_reg)
        mae_weighted_train = mae_train * train_samples
        r2score_val_train = r2_score(data_train_y_reg, predicted_train_reg)
        r2score_weighted_train = r2score_val_train * train_samples
        
        mae_test = mean_absolute_error(data_test_y_reg, predicted_test_reg)
        mae_weighted_test = mae_test * test_samples
        r2score_val_test = r2_score(data_test_y_reg, predicted_test_reg)
        r2score_weighted_test = r2score_val_test * test_samples
        
        mae_weighted_train_list.append(mae_weighted_train)
        mae_weighted_test_list.append(mae_weighted_test)
        r2score_weighted_train_list.append(r2score_weighted_train)
        r2score_weighted_test_list.append(r2score_weighted_test) 

        data_row = [no_of_trees, i, train_samples, test_samples, mae_train, mae_test, r2score_val_train, r2score_val_test]
        writer.writerow(data_row)

       
    total_samples_train = np.sum(train_samples_for_each_fold_list)
    total_samples_test = np.sum(test_samples_for_each_fold_list)
    mae_train_final = np.sum(mae_weighted_train_list)/total_samples_train
    mae_test_final = np.sum(mae_weighted_test_list)/total_samples_test
    r2score_train_final = np.sum(r2score_weighted_train_list)/total_samples_train
    r2score_test_final =  np.sum(r2score_weighted_test_list)/total_samples_test

    data_row = [no_of_trees, "cumulative",total_samples_train, total_samples_test, mae_train_final, mae_test_final, r2score_train_final,r2score_test_final]
    writer.writerow(data_row)

# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()
parser.add_argument('--no_of_trees', dest='no_of_trees', type=str, help='Add no_of_trees')
args = parser.parse_args()
n = args.no_of_trees
no_of_trees = int(n)
print('\nmax_depth : ', max_tree_depth)
print('\nNumber of Trees : ', no_of_trees)
print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))
print('--------------------------------------------------------------------------------------')

print('List of circuits')
train_data_file_reg = '-1000' + data_file_suffix
train_label_file_reg = '-1000' + label_file_suffix
list_ckt_names_reg = capture_ckt_info(train_data_label_path_reg, train_data_file_reg, train_label_file_reg)
list_ckt_names_reg.sort()
#print(list_ckt_names_reg)
        
# K-fold cross validation
# Creating 'k' test sets for k-fold cross validation after shuffling the list
print('\nCreate test sets for 5-fold cross validation')
test_ckts_array = np.array_split(list_ckt_names_reg, k_fold_cross_validation)
print(test_ckts_array)
filename = 'results_' + str(no_of_trees) + '.csv'
with open(filename, 'w', newline='') as fileptr:
    writer = csv.writer(fileptr)
    writer.writerow(header)

    print('  Load Model')
    gb = GradientBoostingRegressor(random_state=0, max_depth=max_tree_depth, n_estimators=no_of_trees)
    print('  Perform '+str(k_fold_cross_validation)+'-fold cross validation and record metrics')
    perform_k_fold_cross_validation(list_ckt_names_reg, k_fold_cross_validation, test_ckts_array, writer, gb)

fileptr.close()
print('\n--------------------------------------------------------------------------------------')
print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))

