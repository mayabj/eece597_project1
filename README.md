This repository contains the scripts and results for the project as captured in the related document.pdf.

The following section provides details of the project directories and files


(1) environment_file.yml : YAML file to recreate the project conda environment

(2) batch_job_commands : Commands to run batch jobs in Jarvis cluster

(3) Find decision tree depth: Script to find optimum depth for small decision trees and related results

(4) Create_Gradboost_classifier_models: Scripts and results for creating Gradientboosting Classifier Models

     1) find_hyperparameter_gradb_class.py : To find optimum hyperparameters
     2) train_gradb_class_save_model.py : To train the models for the chosen hyperparameter values and save the model

(5) Create_Gradboost_regression_models: Scripts and results for creating Gradientboosting Regression Models

      1) find_hyperparameter_gradb_reg.py : To find optimum hyperparameters
      2) train_gradb_reg_save_model.py :  To train the models for the chosen hyperparameter values and save the model

(6) Create_Simple_classifier_models : Scripts to create simple classifier models

(7) Test_the_models : Scripts to test the trained models using Test data

     1) test_gradb_classifier.py : To test GradientBoostingClassifier models for various decision thresholds
     2) test_gradb_reg_as_classifier.py : To test the GradientBoostingRegressor models as a classifier
     3) test_simple_model.py: To test the simple classifier models
