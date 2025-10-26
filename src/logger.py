import logging
import os
from datetime import datetime
LOG_FILE = f"logs/log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), LOG_FILE)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
# log_path already contains the full path to the log file, so just use it
LOG_FILE_PATH = log_path
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
)
if __name__ == "__main__":
    logging.info("Logger has been set up.")
     