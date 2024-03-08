####################################################################################################################################
# This script is used for testing the GradientBoostingClassifier models for various decision thresholds.
# If decision threshold is not a multiple of 50, linear interpolation is performed using closest matching upper and lower models.
####################################################################################################################################

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
model_file_suffix='-1_gradb_class_6d30t'

# Specify File Paths
data_label_path_class = 'project_data/ttnkoi_class_75-975_by_name/'
model_path_class = 'ML_MODELS/ml_models_class_6_30_numpy_1_22/'

decision_points_list = ['75', '125', '175', '225', '275', '325', '375', '425', '475', '525', '575', '625', '675','725','775','825','875','925','975']
#decision_points_list = ['325']
# ===================== Functions =================================================================

def capture_test_data_label_class(data_label_path_class, data_file_class, label_file_class):
    """
    Capture circuit information from Binary Classification data
    @param data_label_path_class:
    @param data_file_class:
    @param label_file_class:
    @return : data_X, data_y
    """
    #print(data_label_path_class)
    regex_data = re.compile(".*({}).*".format(data_file_class))
    regex_label = re.compile(".*({}).*".format(label_file_class))
    list_data_files_class = []
    list_label_files_class = []
    #print(regex_data)
    #print(regex_label)
    # Capture circuit information from binary classification data
    for root, dirs, files in os.walk(data_label_path_class):
        for file in files:
            if regex_data.match(file):
                list_data_files_class.append(file)
            if regex_label.match(file):
                list_label_files_class.append(file)

    no_of_ckts_class = len(list_data_files_class)
    #print('             No of Circuits in Classification data set : ', no_of_ckts_class)

    data_files_list = np.char.add(data_label_path_class, list_data_files_class)
    label_files_list = np.char.add(data_label_path_class, list_label_files_class)
    data_files_list.sort()
    label_files_list.sort()
    #print(data_files_list)
    #print(label_files_list)

    # Concatenate info in csv files
    columns_X = np.genfromtxt(data_files_list[0], delimiter=',', dtype=float).shape[1]
    data_X = np.empty((0, columns_X))
    data_y = np.empty(0)

    for file in data_files_list:
        data_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_X = np.concatenate((data_X, data_X_temp), axis=0)

    for file in label_files_list:
        data_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_y = np.concatenate((data_y, data_y_temp))

    return data_X, data_y


def calculate_metrics(decision_point,data_test_y_class,class_predicted):
    """
    calculate metrics
    @param decision_point:
    @param model_file_suffix:
    @param data_file_suffix:
    @param label_file_suffix:
    @return : data_test_y_class, class_predicted
    """
    print('Calculate metrics')
    mean_accuracy = accuracy_score(data_test_y_class, class_predicted)
    tn, fp, fn, tp = confusion_matrix(data_test_y_class, class_predicted).ravel()
    mcc = matthews_corrcoef(data_test_y_class, class_predicted)
    print(header)
    print(str(decision_point) +','+ str(mean_accuracy) +','+ str(tp) +','+ str(tn) +','+ str(fp) +','+ str(fn) +','+ str(mcc))

def predict_proba(decision_point,reference_decision_point, model_file_suffix,data_file_suffix,label_file_suffix):
    """
    predict probabilities
    @param decision_point:
    @param threshold_point:
    @param model_file_suffix:
    @param data_file_suffix:
    @param label_file_suffix:
    @return : data_test_y_class, class_predicted
    """
    model_file_name_class = 'mvto-' + str(reference_decision_point) + model_file_suffix
    model_file_path_class = str(np.char.add(model_path_class, model_file_name_class))
    print('Model name : ', model_file_path_class)
    #print('Load Classification Model')
    model_load_var_class = joblib.load(model_file_path_class)
    #print('Load Classification Data')
    data_file_class = '-' + str(decision_point) + data_file_suffix
    label_file_class = '-' + str(decision_point) + label_file_suffix
    data_test_X_class, data_test_y_class = capture_test_data_label_class(data_label_path_class, data_file_class, label_file_class)
    #print('Predict class')
    #class_predicted = model_load_var_class.predict(data_test_X_class).round()
    proba_predicted = model_load_var_class.predict_proba(data_test_X_class)[:, [1]]
    return data_test_y_class, proba_predicted

# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))
for d in decision_points_list:
    decision_point = int(d)
    print('\nDecision point : ', decision_point)
    if (decision_point % 50 == 0):
        data_test_y_class, predicted_proba=predict_proba(decision_point,decision_point, model_file_suffix,data_file_suffix,label_file_suffix)
        class_predicted=predicted_proba
    else:
        #print("Need to interpolate")
        lower_point=int(50 * (int(decision_point/50)))
        upper_point=int(50 * ((int(decision_point/50))+1))
        distance_low=decision_point-lower_point
        distance_high=upper_point-decision_point
        print('distances :', distance_low, distance_high)
        data_test_y_class, predicted_proba_lower = predict_proba(decision_point,lower_point, model_file_suffix, data_file_suffix,label_file_suffix)
        data_test_y_class, predicted_proba_upper = predict_proba(decision_point,upper_point, model_file_suffix, data_file_suffix, label_file_suffix)
        #class_predicted_lower.resize(class_predicted_upper.shape)
        class_predicted=(distance_high*predicted_proba_lower + distance_low*predicted_proba_upper)/(distance_high+distance_low)


    # Calculate Metrics
    class_predicted = [0 if val < 0.5 else 1 for val in class_predicted]
    calculate_metrics(decision_point, data_test_y_class, class_predicted)

print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))



