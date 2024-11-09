import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.model_evaluation import Evaluation


@dataclass
class EvaluationPipeline:
    def __init__(self):
        pass

    def initiate_pipeline(self):
        try:
            evaluation_obj = Evaluation()
            evaluation_obj.initiate_evaluation()
            evaluation_obj.save_score()


        except Exception as e:
            logging.info("Error in main method")
            raise CustomException(e,sys)

if __name__ == "__main__":
    pipeline_obj = EvaluationPipeline()
    pipeline_obj.initiate_pipeline()