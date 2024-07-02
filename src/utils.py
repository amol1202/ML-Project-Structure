import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from pathlib import Path
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        os.makedirs(Path(file_path).parent,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            model_score = r2_score(y_test , y_pred)

            report[list(models.keys())[i]] = model_score
        
        return report
    except Exception as e:
        raise CustomException(e,sys)
