import sys
import os
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"../"))
sys.path.append(MAIN_DIR)

from utils.exception import CustomException
from utils.logger import logging
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import DistilBertTokenizer
import torch 

class Data:
    def __init__(self,dataset_path, test_size,max_length):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.max_length = max_length
    def read_data(self):
        try:
            df = pd.read_csv(self.dataset_path)
            logging.info(f"Read data from{self.dataset_path}")
            X= list(df['news'])
            y= list(df['sentiment'])
            return train_test_split(X, y, test_size=self.test_size,stratify=y)
        except Exception as e:
            raise CustomException(e,sys)
        
    def data_tokenizetion(self,X_train, X_val):
        try:
           tokenizer_URL ="distilbert-base-uncased"
           tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_URL)
           X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=self.max_length)
           X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=self.max_length)
           logging.info("Data tokenized successfully")
           return X_train_tokenized, X_val_tokenized
        except Exception as e:
            raise CustomException(e,sys)
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

