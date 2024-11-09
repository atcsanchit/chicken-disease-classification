import sys
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from pathlib import Path
import time
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.prepare_callback import PrepareCallback


@dataclass
class TrainingConfig:
    updated_base_model_path = "artifacts/prepare_base_model/base_model_updated.h5"
    params_image_size = [224,224,3]
    params_batch_size = 16
    training_data = os.path.join("artifacts/data_ingestion","Chicken-fecal-images")
    params_epochs = 5
    params_is_augmentation = True
    trained_model_path = "artifacts/training/model.h5"
    params_learning_rate = 0.01

class Training:
    def __init__(self):
        self.training_config = TrainingConfig
    
    def get_base_model(self):
        try:
            # base_model = load_model(self.training_config.updated_base_model_path, compile=False)
            # inputs = tf.keras.Input(shape=(224, 224, 3))
            # outputs = base_model(inputs)
            # self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            # self.model.compile(
            #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            #     loss='categorical_crossentropy',
            #     metrics=['accuracy']
            # )


            self.model = tf.keras.models.load_model(
                self.training_config.updated_base_model_path,
                compile = False
            )
            self.model.compile(
                optimizer = tf.keras.optimizers.SGD(learning_rate = self.training_config.params_learning_rate),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"]
            )

        except Exception as e:
            logging.info("Error in get_base_model")
            raise CustomException(e,sys)

    def train_valid_generator(self):
        try: 
            datagenerator_kwargs = dict(
                rescale = 1./255,
                validation_split = 0.20
            )

            dataflow_kwargs = dict(
                target_size = self.training_config.params_image_size[:-1],
                batch_size = self.training_config.params_batch_size,
                interpolation = "bilinear"
            )

            valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )

            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory = self.training_config.training_data,
                subset = "validation",
                shuffle = False,
                **dataflow_kwargs
            )

            if self.training_config.params_is_augmentation:
                train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range = 40,
                    horizontal_flip = True,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    **datagenerator_kwargs
                )

            else:
                train_datagenerator = valid_datagenerator

            self.train_generator = train_datagenerator.flow_from_directory(
                directory = self.training_config.training_data,
                subset = "training",
                shuffle = True,
                **dataflow_kwargs
            )
        
        except Exception as e:
            logging.info("Error in train_valid_generator")
            raise CustomException(e,sys)

    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        try:
            model.save(path)
        
        except Exception as e:
            logging.info("Error in save_model")
            raise CustomException(e,sys)
        

    def initiate_train(self, callback_list:list):
        try:            
            self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
            self.validation_steps = self.valid_generator.samples // self.train_generator.batch_size

            self.model.fit(
                self.train_generator,
                epochs = self.training_config.params_epochs,
                steps_per_epoch = self.steps_per_epoch,
                validation_steps = self.validation_steps,
                validation_data = self.valid_generator,
                callbacks=callback_list
            )

            self.save_model(
                path = self.training_config.trained_model_path,
                model = self.model
            )
        
        except Exception as e:
            logging.info("Error in save_model")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    callback_obj = PrepareCallback()
    callback_list = callback_obj.get_tb_ckpt_callbacks()
    training_obj = Training()
    training_obj.get_base_model()
    training_obj.train_valid_generator()
    training_obj.initiate_train(
        callback_list=callback_list
    )