import sys
import os
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"../"))
sys.path.append(MAIN_DIR)

import kagglehub
import pandas as pd 
from utils.logger import logging
from utils.exception import CustomException


class LoadFromKaggle:
    def load(self,url, file_name ):
        try:
            path = kagglehub.dataset_download(url)
            pd.read_csv(path+"/news.csv").to_csv(f"../../data/{file_name}")
            logging.info(f"Read data to {file_name}")
        except Exception as e:
             raise CustomException(e,sys)

if __name__=="__main__":
    obj = LoadFromKaggle()
    PATH_OF_DATA = "myrios/news-sentiment-analysis"
    FILE_NAME = "data.csv"
    obj.load(url=PATH_OF_DATA, file_name=FILE_NAME)
            
 


