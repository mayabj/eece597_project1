#################################################################################################
# This script is used for testing the GradientBoostingRegressor models as a classifier.
# Here, the trained regression model 'mvto-1000-1_gradb_reg_3d100t' is first used to predict
# the remaining no: of iterations using which the absolute no: of iterations is calculated.
# This value is converted into a classification prediction and compared with the ground truth.
#################################################################################################

# Import packages

import time
import joblib
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

# Declare Variables
header=["Decision Point", "Accuracy", "True Positive", "True Negative","False Positive", "False Negative", "MCC"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
reg_model_file_name='mvto-1000-1_gradb_reg_3d100t'

# Specify File Paths
data_label_path_class = 'project_data/ttnkoi_class_75-975_by_name/'
model_path_reg = 'ML_MODELS/ml_models_reg_depth_3/'
decision_points_list = ['75', '125', '175', '225', '275', '325', '375', '425', '475', '525', '575', '625', '675','725','775','825','875','925','975']

# ===================== Functions ===============================================   ==================

def capture_test_data_label_class(data_label_path_class, data_file_class, label_file_class):
    """
    Capture circuit information from Binary Classification data
    @param data_label_path_class:
    @param data_file_class:
    @param label_file_class:
    @return : data_train_X, data_train_y
    """

    regex_data = re.compile(".*({}).*".format(data_file_class))
    regex_label = re.compile(".*({}).*".format(label_file_class))
    list_data_files_class = []
    list_label_files_class = []
    # Capture circuit information from binary classification data
    for root, dirs, files in os.walk(data_label_path_class):
        for file in files:
            if regex_data.match(file):
                list_data_files_class.append(file)
            if regex_label.match(file):
                list_label_files_class.append(file)

    #no_of_ckts_class = len(list_data_files_class)
    #print('             No of Circuits in Classification data set : ', no_of_ckts_class)

    data_files_list = np.char.add(data_label_path_class, list_data_files_class)
    label_files_list = np.char.add(data_label_path_class, list_label_files_class)
    data_files_list.sort()
    label_files_list.sort()
    #print(data_files_list)
    #print(label_files_list)

    # Concatenate info in csv files
    columns_X = np.genfromtxt(data_files_list[0], delimiter=',', dtype=float).shape[1]
    data_train_X = np.empty((0, columns_X))
    data_train_y = np.empty(0)

    for file in data_files_list:
        data_train_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_train_X = np.concatenate((data_train_X, data_train_X_temp), axis=0)

    for file in label_files_list:
        data_train_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_train_y = np.concatenate((data_train_y, data_train_y_temp))

    return data_train_X, data_train_y


# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))
model_file_path_reg = str(np.char.add(model_path_reg, reg_model_file_name))
print('Model name : ', model_file_path_reg)

for d in decision_points_list:
    decision_point = int(d)
    print('\nDecision point : ', decision_point)

    print('    Load Regression Model')
    model_load_var_reg = joblib.load(model_file_path_reg)

    print('    Load Classification Data')
    data_file_class = '-' + str(decision_point) + data_file_suffix
    label_file_class = '-' + str(decision_point) + label_file_suffix
    data_test_X_class, data_test_y_class = capture_test_data_label_class(data_label_path_class, data_file_class, label_file_class)

    print('    Iteration prediction with regressor model')
    y_remaining_predicted_with_regressor = model_load_var_reg.predict(data_test_X_class).round()
    #np.savetxt('iteration_predicted_with_regressor.txt', y_remaining_predicted_with_regressor)
    data_test_X_class_column_0=data_test_X_class[:,0]
    y_absolute_prediction_with_regressor = y_remaining_predicted_with_regressor + data_test_X_class_column_0

    print('    Translate the regression prediction to classification')
    class_prediction_from_y_absolute=y_absolute_prediction_with_regressor
    class_prediction_from_y_absolute = [0 if val > decision_point else 1 for val in class_prediction_from_y_absolute]

    print('    Calculate metrics')
    mean_accuracy = accuracy_score(data_test_y_class, class_prediction_from_y_absolute)
    tn, fp, fn, tp = confusion_matrix(data_test_y_class, class_prediction_from_y_absolute).ravel()
    mcc = matthews_corrcoef(data_test_y_class, class_prediction_from_y_absolute)
    print(header)
    print(str(decision_point) +','+ str(mean_accuracy) +','+ str(tp) +','+ str(tn) +','+ str(fp) +','+ str(fn) +','+ str(mcc))

print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))


