import sys
import os
from utils.exception import CustomException
from utils.logger import logging

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction,DistilBertTokenizer
from huggingface_hub import login
from data_preparation import Data, Dataset

login(token="ENTER YOUR TOKEN")


MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

DATASET_PATH = "../../data/cleaned_data.csv"
TEST_SIZE = 0.4
MAX_LENGTH = 512
OUTPUT_DIR = "ENTER YOUR OUTPUT DIR"
NUM_EPOCHS = 1
BATCH_SIZE = 32
MODEL_URL = "distilbert-base-uncased"

class Train:
    def __init__(self):
        data = Data(dataset_path=DATASET_PATH, test_size=TEST_SIZE, max_length=MAX_LENGTH)
        self.X_train, self.X_val, self.y_train, self.y_val = data.read_data()
        self.X_train_tokenized, self.X_val_tokenized = data.data_tokenizetion(self.X_train, self.X_val)
        self.train_dataset = Dataset(self.X_train_tokenized, self.y_train)
        self.val_dataset = Dataset(self.X_val_tokenized, self.y_val)
        logging.info("Train object created")

    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        recall = recall_score(y_true=labels, y_pred=predictions)
        precision = precision_score(y_true=labels, y_pred=predictions)
        f1 = f1_score(y_true=labels, y_pred=predictions)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def model_load(self, model_url, num_labels):
        return DistilBertForSequenceClassification.from_pretrained(model_url, num_labels=num_labels), DistilBertTokenizer.from_pretrained(model_url)

    def train_model(self):
        model, tokenizer = self.model_load(model_url=MODEL_URL,num_labels=2)
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            evaluation_strategy="epoch",
        )
     
        trainer = Trainer(
            model=model,
            tokenizer= tokenizer,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics, 
        )

        try:

            trainer.train()
            trainer.evaluate()
            trainer.model.config.id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
            trainer.model.config.id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
            # push to huggingface hub 
            trainer.push_to_hub("CustomModel")
            logging.info("Model saved to hub")
        except Exception as e:
            raise CustomException(e,sys)
            


if __name__ == "__main__":
    train = Train()
    train.train_model()
