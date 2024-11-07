import sys
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class PrepareBaseModelConfig:
    
    base_model_path = "../artifacts/prepare_base_model/base_model.h5"
    updated_base_model_path = "../artifacts/prepare_base_model/base_model_updated.h5"
    params_image_shape = [224, 224, 3]
    params_learning_rate = 0.01
    params_include_top = False
    params_weights = "imagenet"
    params_classes = 2

class PrepareBaseModel:
    def __init__(self):
        self.prepare_base_model_config_obj = PrepareBaseModelConfig()
    
    def get_base_model(self):
        try:
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape = self.prepare_base_model_config_obj.params_image_shape,
                weights = self.prepare_base_model_config_obj.params_weights,
                include_top = self.prepare_base_model_config_obj.params_include_top
            )
            self.save_model(path=self.prepare_base_model_config_obj.base_model_path, model=self.model)
        except Exception as e:
            logging.info("Error in get_base_model")
            raise CustomException(e,sys)
        

    # @staticmethod
    def prepare_full_model(self, model, classes, freeze_all, freeze_till, learning_rate):
        try:
            if freeze_all:
                for layer in model.layers:
                    model.trainable = False
            elif (freeze_till is not None) and (freeze_till > 0):
                for layer in model.layers[:-freeze_till]:
                    model.trainable = False

            flatten_in = tf.keras.layers.Flatten()(model.output)
            prediction = tf.keras.layers.Dense(
                units = classes,
                activation = "softmax"
            )(flatten_in)

            full_model = tf.keras.models.Model(
                inputs = model.input,
                outputs = prediction
            )

            full_model.compile(
                optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"]
            )

            full_model.summary()
            return full_model
        
        except Exception as e:
            logging.info("Error in prepare_full_model")
            raise CustomException(e,sys)
        


    def update_base_model(self):
        try:
            self.full_model = self.prepare_full_model(
                model = self.model,
                classes = self.prepare_base_model_config_obj.params_classes,
                freeze_all = True,
                freeze_till = None,
                learning_rate = self.prepare_base_model_config_obj.params_learning_rate
            )
            self.save_model(path = self.prepare_base_model_config_obj.updated_base_model_path, model = self.full_model)
        
        except Exception as e:
            logging.info("Error in update_base_model")
            raise CustomException(e,sys)
        
    # @staticmethod
    def save_model(self, path:Path, model: tf.keras.Model):
        try:
            model.save(path)
        
        except Exception as e:
            logging.info("Error in update_base_model")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    prepare_base_model_obj = PrepareBaseModel()
    prepare_base_model_obj.get_base_model()
    prepare_base_model_obj.update_base_model()