import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
import os
from src.utils import save_object

@dataclass
class dataTransformationConfig:
    preprocessor_obj_filepath = os.path.join("artifact", "preprocessor.pkl")

class dataTransformation:
    def __init__(self):
        self.data_transformation_config = dataTransformationConfig()
    
    def get_dataTransformer_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline,numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_dataTransformer_obj()
            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            inputFeature_train = train_df.drop([target_column],axis=1)
            targetFeature_train = train_df[target_column]

            inputFeature_test = test_df.drop([target_column],axis=1)
            targetFeature_test = test_df[target_column]

            inputFeature_train_arr = preprocessing_obj.fit_transform(inputFeature_train)
            inputFeature_test_arr = preprocessing_obj.transform(inputFeature_test)
            
            train_arr = np.c_[inputFeature_train_arr, targetFeature_train]
            test_arr = np.c_[inputFeature_test_arr, targetFeature_test]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_filepath,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )

        except Exception as e:
            raise CustomException(e,sys)