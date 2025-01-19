# import logging
# import os

# from from_root import from_root
# from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# log_dir = 'logs'

# logs_path = os.path.join(from_root(), log_dir, LOG_FILE)

# os.makedirs(log_dir, exist_ok=True)


# logging.basicConfig(
#     filename=logs_path,
#     format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
#     level=logging.DEBUG,
# )


import logging
import os
from from_root import from_root
from datetime import datetime

# Define log file name and directory
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = os.path.join(from_root(), 'logs')  # Ensure the directory is rooted correctly

# Create the logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Complete path for the log file
logs_path = os.path.join(log_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

# Example usage
logger = logging.getLogger(__name__)
logger.info("Logging is set up correctly.")
