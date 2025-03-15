import sys
import os
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"../"))
sys.path.append(MAIN_DIR)
import pandas as pd 
from utils.logger import logging
from utils.exception import CustomException
class DataTransformation:
    def extract(self,url, file_name,features):
        try:
            df=pd.read_csv(url)
            df= df[features]
            df.drop_duplicates()
            logging.info("Delete duplicates")
            df.dropna()
            df['sentiment'] = df['sentiment'].map({'NEGATIVE':0,'POSITIVE':1})
            df.to_csv(f"../../data/{file_name}")
            logging.info(f"Cleaned data saved in{file_name}")
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataTransformation()
    DATA_PATH = "../../data/data.csv"
    FILE_NAME = "cleaned_data.csv"
    FEATURES = ['news','sentiment']
    obj.extract(url=DATA_PATH,file_name=FILE_NAME,features=FEATURES)
 
     
