import sys
import os
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"../"))
sys.path.append(MAIN_DIR)

import logging
import logging.config
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join("../../","logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)
LOGS_FILE_PATH = os.path.join(logs_path,LOG_FILE)
logging.basicConfig(
    filename=LOGS_FILE_PATH,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)

