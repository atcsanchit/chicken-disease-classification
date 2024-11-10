#script to execute all the pipelines at once from scratch
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.prepare_base_model_pipeline import PrepareBaseModelPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.evaluation_pipeline import EvaluationPipeline


@dataclass
class Pipeline:
    def __init__(self):
        pass

    def execute_pipeline(self):
        try:
            data_ingestion_obj = DataIngestionPipeline()
            data_ingestion_obj.initiate_pipeline()

            prepare_base_model_obj = PrepareBaseModelPipeline()
            prepare_base_model_obj.initiate_pipeline()

            training_obj = TrainingPipeline()
            training_obj.initiate_pipeline()

            evaluation_obj = EvaluationPipeline()
            evaluation_obj.initiate_pipeline()

            print("all pipelines are successfully executed")
            logging.info("all pipelines are successfully executed")

        except Exception as e:
            logging.info("Error in execute_pipeline method in main strategy")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    pipeline_obj = Pipeline()
    pipeline_obj.execute_pipeline()