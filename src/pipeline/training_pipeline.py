import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.prepare_callback import PrepareCallback
from src.components.training import Training


@dataclass
class TrainingPipeline:
    def __init__(self):
        pass

    def initiate_pipeline(self):
        try:
            logging.info("Initiating training pipeline")
            print("Initiating training pipeline")
            prepare_callback_obj = PrepareCallback()
            callback_list = prepare_callback_obj.get_tb_ckpt_callbacks()     
            training_obj = Training()
            training_obj.get_base_model()
            training_obj.train_valid_generator()
            training_obj.initiate_train(
                callback_list=callback_list
            )
            print("training pipeline has been successfully executed")
            print("*"*20)


        except Exception as e:
            logging.info("Error in main method")
            raise CustomException(e,sys)

if __name__ == "__main__":
    pipeline_obj = TrainingPipeline()
    pipeline_obj.initiate_pipeline()