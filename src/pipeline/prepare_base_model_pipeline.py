import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.prepare_base_model import PrepareBaseModel


@dataclass
class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def initiate_pipeline(self):
        try:
            prepare_base_model_obj = PrepareBaseModel()
            prepare_base_model_obj.get_base_model()
            prepare_base_model_obj.update_base_model()      

        except Exception as e:
            logging.info("Error in main method")
            raise CustomException(e,sys)

if __name__ == "__main__":
    pipeline_obj = PrepareBaseModelPipeline()
    pipeline_obj.initiate_pipeline()