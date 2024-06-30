import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.components.data_transformation import dataTransformation

@dataclass
class dataIngestionConfig:
    train_data_path:str= os.path.join('artifact', "train.csv")    
    test_data_path:str= os.path.join('artifact', "test.csv")
    raw_data_path:str= os.path.join('artifact', "raw.csv")

class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as a dataframe")

            
            os.makedirs(Path(self.ingestion_config.train_data_path).parent,exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion completed!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            CustomException(e,sys)