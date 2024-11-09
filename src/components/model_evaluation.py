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

    @staticmethod
    def load_model(path:Path) -> tf.keras.Model:
        return tf.keras.models.load_model(
            path
        )

    def initiate_evaluation(self):
        self.model = self.load_model(self.evaluation_config.path_to_model)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',  # or whatever loss you used during training
            metrics=[tf.keras.metrics.Accuracy()]
        )
        self.valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {"loss":self.score[0], "accuracy":self.score[1]}
        save_json(path = Path("scores.json"), data = scores)


if __name__ == "__main__":
    evaluation_obj = Evaluation()
    evaluation_obj.initiate_evaluation()
    evaluation_obj.save_score()