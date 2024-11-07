import sys
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class PrepareCallbackConfig:
    tensorboard_root_log_dir = "artifacts/prepare_callbacks/tensorboard_log_dir"
    checkpoint_model_filepath = "artifacts/prepare_callbacks/checkpoint_dir/model.keras"

class PrepareCallback:
    def __init__(self):
        self.callback_config_obj = PrepareCallbackConfig()
    
    @property
    def create_tb_callbacks(self):
        try:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            tb_running_log_dir = os.path.join(
                self.callback_config_obj.tensorboard_root_log_dir,
                f"tb_logs_at_{timestamp}"
            )
            return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
        except Exception as e:
            logging.info("Error in create_tb_callbacks")
            raise CustomException(e, sys)
        
    @property
    def create_ckpt_callbacks(self):
        try:
            return tf.keras.callbacks.ModelCheckpoint(
                filepath = self.callback_config_obj.checkpoint_model_filepath,
                save_best_only = True
            )
 
        except Exception as e:
            logging.info("Error in create_tb_callbacks")
            raise CustomException(e,sys)
        
    def get_tb_ckpt_callbacks(self):
        try:
            return [
                self.create_tb_callbacks,
                self.create_ckpt_callbacks
            ]
        except Exception as e:
            logging.info("Error in get_tb_ckpt_callbacks")
            raise CustomException(e,sys)

if __name__ == "__main__":
    prepare_callback_obj = PrepareCallback()
    prepare_callback_obj.get_tb_ckpt_callbacks()