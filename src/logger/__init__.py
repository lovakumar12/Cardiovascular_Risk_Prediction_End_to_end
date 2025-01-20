# import logging
# import os

# from from_root import from_root
# from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# log_dir = 'logs'

# logs_path = os.path.join(from_root(), log_dir, LOG_FILE)
# #os.makedirs(os.path.dirname(logs_path), exist_ok=True)


# os.makedirs(log_dir, exist_ok=True)


# logging.basicConfig(
#     filename=logs_path,
#     format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
#     level=logging.DEBUG,
# )



import logging
import os
from datetime import datetime
from from_root import from_root

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = 'logs'
#logs_path = os.path.join(from_root(), log_dir, LOG_FILE)
logs_path=r"C:\Users\kumar\Cardiovascular_Risk_Prediction_End_to_end\logs\test_log.log"

# Ensure the parent directories are created
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

# Add console logging as well
console_handler = logging.StreamHandler()
logging.getLogger().addHandler(console_handler)

# Example log statement to check
logging.debug("This is a debug message.")
