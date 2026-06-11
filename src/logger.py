import os
import logging
from datetime import datetime

# 1. Create a unique, timestamped file name for the current run
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Define the path where the logs directory should sit (root_dir/logs/)
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# 3. Create the 'logs' folder if it doesn't already exist
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

# 4. Set up the absolute configuration path
LOG_FILE_PATH = logs_path

# 5. Configure the logging system
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)