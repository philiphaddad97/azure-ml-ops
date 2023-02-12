#!/bin/bash
az ml model register -g $(azureml.resourceGroup) -w $(azureml.workspaceName) -n svm --model-path svc.pkl -d "SVM Classification Model to Identify Keystrokes user"
az ml model register -g $(azureml.resourceGroup) -w $(azureml.workspaceName) -n xgboost --model-path xgb_classifier.pkl -d "XGBoost Classification Model to Identify Keystrokes user"
az ml model register -g $(azureml.resourceGroup) -w $(azureml.workspaceName) -n rf --model-path random_forest_classifier.pkl -d "Random Forest Classification Model to Identify Keystrokes user"