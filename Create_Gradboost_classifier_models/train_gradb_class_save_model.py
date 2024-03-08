#############################################################################################################
# This script trains GradientBoosting Classifier models for given hyperparameter values and saves the model
##############################################################################################################

# Import packages

import argparse
import joblib
import numpy as np
import os
import re
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

# Declare Variables
header=["Decision Point", "Max depth", "No of Trees", "Accuracy (Train)", "True Positive (Train)",
        "True Negative(Train)","False Positive (Train)", "False Negative (Train)", "MCC (Train)"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
k_fold_cross_validation = int(5) ## 5 fold cross validation
max_tree_depth_list = ['6']
no_of_trees_list = ['30']

# Specify File Paths
train_data_file_path_reg = 'G:/ProjectData/routing_data_train/mvto-1000-1_train_X.csv'
train_label_file_path_reg = 'G:/ProjectData/routing_data_train/mvto-1000-1_train_y.csv'
train_data_label_path_class = 'G:/ProjectData/circuit_split_train_data_for_cv/'

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
    print("             Number of routing runs in train data : ", number_of_runs_train)
    return data_train_X,data_train_y


def capture_ckt_info_class(train_data_label_path_class, train_data_file_class, train_label_file_class):
    """
    Capture circuit information from Binary Classification data
    @param train_data_label_path_class:
    @param train_data_file_class:
    @param train_label_file_class:
    @return : list_ckt_names_class
    """

    # regex_data = re.compile('.*(%s).*'%train_data_file_class)
    # regex_label = re.compile('.*(%s).*'%train_label_file_class)
    regex_data = re.compile(".*({}).*".format(train_data_file_class))
    regex_label = re.compile(".*({}).*".format(train_label_file_class))
    list_data_files_class = []
    list_label_files_class = []
    list_ckt_names_class = []
    # Capture circuit information from binary classification data
    for root, dirs, files in os.walk(train_data_label_path_class):
        for file in files:
            if regex_data.match(file):
                list_data_files_class.append(file)
                list_ckt_names_class.append(file.split(train_data_file_class)[0])
            if regex_label.match(file):
                list_label_files_class.append(file)

    no_of_ckts_class = len(list_ckt_names_class)
    print('             No of Circuits in Classification data set : ', no_of_ckts_class)
    #print('Circuit Names :')
    #pp = pprint.PrettyPrinter(width=100, compact=True)
    #pp.pprint(list_ckt_names_class)
    return list_ckt_names_class


def organize_train_data(list_ckt_names_class,train_data_label_path_class):
    """
    Organize Binary Classification data
    @param list_ckt_names_class:
    @param train_data_label_path_class
    @return : data_train_X,data_train_y
    """
    train_set = list_ckt_names_class
    train_file_root_list = np.char.add(train_data_label_path_class, train_set)
    train_data_file_list = np.char.add(train_file_root_list, train_data_file_class)
    train_label_file_list = np.char.add(train_file_root_list, train_label_file_class)

    # Concatenate info in csv files
    data_train_X = np.empty((0, columns_X))
    data_train_y = np.empty(0)

    for file in train_data_file_list:
        data_train_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_train_X = np.concatenate((data_train_X, data_train_X_temp), axis=0)

    for file in train_label_file_list:
        data_train_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_train_y = np.concatenate((data_train_y, data_train_y_temp))

    return data_train_X, data_train_y



# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++
now = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--decision_point', dest='decision_point', type=str, help='Add decision_point')
args = parser.parse_args()
decision_point = args.decision_point
print('\nDecision point : ', decision_point)
print('Max Depth values :', max_tree_depth_list)
print('No: of trees (n_estimators) : ', no_of_trees_list)
#print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))
print('\nProgram Start Time : ',now.strftime("%Y-%m-%d %H:%M:%S"))
print('--------------------------------------------------------------------------------------')
print('\nLoad Regression Data : Train')
(data_train_X_reg, data_train_y_reg) = load_train_data_reg(train_data_file_path_reg, train_label_file_path_reg)
columns_X = data_train_X_reg.shape[1]  # no: of features

print('\nCapture Binary Classification Data')
train_data_file_class = '-' + str(decision_point) + data_file_suffix
train_label_file_class = '-' + str(decision_point) + label_file_suffix
list_ckt_names_class = capture_ckt_info_class(train_data_label_path_class, train_data_file_class, train_label_file_class)
list_ckt_names_class.sort()
print(list_ckt_names_class)
        
for m in max_tree_depth_list:
    max_tree_depth = int(m)
    for n in no_of_trees_list:
        no_of_trees = int(n)
        print('\nmax_depth : ', max_tree_depth)
        print('No: of trees (n_estimators) : ', no_of_trees)

        print('  Load Classification Model')
        gb = GradientBoostingClassifier(random_state=0, max_depth=max_tree_depth, n_estimators=no_of_trees)
 
        print('  Fit model to classification data')
        # Fit model to classification data
        data_train_X_class, data_train_y_class = organize_train_data(list_ckt_names_class, train_data_label_path_class)
        clf = gb.fit(data_train_X_class, data_train_y_class)

        #Calculate Metrics
        print('   Calculate metrics for Train data')
        mean_accuracy_train = clf.score(data_train_X_class, data_train_y_class)
        predicted_class_train = gb.predict(data_train_X_class)
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(data_train_y_class, predicted_class_train).ravel()
        mcc_train = matthews_corrcoef(data_train_y_class, predicted_class_train)
        print(header)
        print(str(decision_point) +','+ str(max_tree_depth)+','+ str(no_of_trees) +','+ str(mean_accuracy_train) +','+ str(tp_train) +','+ str(tn_train) +','+ str(fp_train) +','+ str(fn_train) +','+ str(mcc_train))
        print('    Save the model : '+'mvto-' + str(decision_point) + '-1_gradb_class_' + str(max_tree_depth) +'d' + str(no_of_trees) + 't')
        save_file_name = 'mvto-' + str(decision_point) + '-1_gradb_class_' + str(max_tree_depth) +'d' + str(no_of_trees) + 't'
        joblib.dump(clf, save_file_name)

print('\n--------------------------------------------------------------------------------------')
#print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))
now = datetime.datetime.now()
print('\nProgram End Time : ',now.strftime("%Y-%m-%d %H:%M:%S"))

