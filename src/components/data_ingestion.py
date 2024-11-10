import sys
import os
import urllib.request as request
import zipfile
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.local_data_file = "artifacts/data_ingestion/data.zip"
        self.source_URL = "https://github.com/entbappy/Branching-tutorial/raw/master/Chicken-fecal-images.zip"
        self.unzip_dir = "artifacts/data_ingestion"

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config_obj = DataIngestionConfig()

    def download_file(self):
        try:
            if not os.path.exists(self.data_ingestion_config_obj.local_data_file):
                os.makedirs(os.path.dirname(self.data_ingestion_config_obj.local_data_file), exist_ok=True)
                filename, headers = request.urlretrieve(
                    url=self.data_ingestion_config_obj.source_URL,
                    filename=self.data_ingestion_config_obj.local_data_file
                )
                logging.info("Executed download_file")
            else:
                logging.info("Data ingestion file exist")
        except Exception as e:
            logging.info("Error in download_file method")
            raise CustomException(e,sys)

    def extract_zip_file(self):
        try:
            unzip_path = self.data_ingestion_config_obj.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.data_ingestion_config_obj.local_data_file,'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logging.info("Executed extract_zip_file")
        except Exception as e:
            logging.info("Error in extract_zip_file")
            raise CustomException(e,sys)
        
    
    def initiate_data_ingestion(self):
        try:
            data_ingestion_obj = DataIngestion()
            data_ingestion_obj.download_file()
            data_ingestion_obj.extract_zip_file()
        except Exception as e:
            logging.info("Error in initiate_data_ingestion")
            raise CustomException(e,sys)


if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()




