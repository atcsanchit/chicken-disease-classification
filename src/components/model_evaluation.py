import sys
import os
from dataclasses import dataclass
import tensorflow as tf
from pathlib import Path

from src.logger import logging
from src.exception import CustomException
from src.utils.common import save_json


@dataclass
class EvaluationConfig:
    path_to_model = "artifacts/training/model.h5"
    training_data = os.path.join("artifacts/data_ingestion","Chicken-fecal-images")
    params_image_size = [224,224,3]
    params_batch_size = 16

class Evaluation:
    def __init__(self):
        self.evaluation_config = EvaluationConfig()
    
    def valid_generator(self):
        try:    
            datagenerator_kwargs = dict(
                rescale = 1./255,
                validation_split = 0.20
            )

            dataflow_kwargs = dict(
                target_size = self.evaluation_config.params_image_size[:-1],
                batch_size = self.evaluation_config.params_batch_size,
                interpolation = "bilinear"
            )

            valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )

            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory = self.evaluation_config.training_data,
                subset = "validation",
                shuffle = False,
                **dataflow_kwargs
            )
        
        except Exception as e:
            logging.info("Error in valid_generator")
            raise CustomException(e,sys)

    @staticmethod
    def load_model(path:Path) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(
                path
            )
        
        except Exception as e:
            logging.info("Error in load_model")
            raise CustomException(e,sys)

    def initiate_evaluation(self):
        try:                
            self.model = self.load_model(self.evaluation_config.path_to_model)
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',  # or whatever loss you used during training
                metrics=[tf.keras.metrics.Accuracy()]
            )
            self.valid_generator()
            self.score = self.model.evaluate(self.valid_generator)
        
        except Exception as e:
            logging.info("Error in initiate_evaluation")
            raise CustomException(e,sys)

    def save_score(self):
        try:
            scores = {"loss":self.score[0], "accuracy":self.score[1]}
            save_json(path = Path("scores.json"), data = scores)

        except Exception as e:
            logging.info("Error in save_score")
            raise CustomException(e,sys)


if __name__ == "__main__":
    evaluation_obj = Evaluation()
    evaluation_obj.initiate_evaluation()
    evaluation_obj.save_score()