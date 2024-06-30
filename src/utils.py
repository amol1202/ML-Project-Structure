import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from pathlib import Path

def save_object(file_path,obj):
    try:
        os.makedirs(Path(file_path).parent,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)