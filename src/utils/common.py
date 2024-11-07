import sys
import os
# from box.exceptions import BoxValueError
import yaml
from src.logger import logging
from src.exception import CustomException
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path:Path)-> ConfigBox:
    try:
        with open(file=path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
        
    except Exception as e:
        logging.info("Error in read_yaml utils")
        raise CustomException(e,sys)


@ensure_annotations
def create_directory(paths:list, verbose=True):
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logging.info(f"created directory at :{path}")

    except Exception as e:
        logging.info("Error in create_directory utils")
        raise CustomException(e,sys)
