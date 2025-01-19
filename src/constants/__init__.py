# code part  writing in  constant file
import os
from datetime import date
from dotenv import load_dotenv
load_dotenv()

DATABASE_NAME="Cardeo_vascular_risk_DB"

COLLECTION_NAME="Cardeo_data"

MONGODB_URL_KEY="MONGODB_URL"

PIPELINE_NAME:str ="Demo_project"  # NOT A SRC its pipeline name

ARTIFICAT_DIR:str="artificat"

MODEL_FILE_NAME="model.pkl"

TARGET_COLUMN="TenYearCHD"

PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"


FILE_NAME:str="Cardeo_vasular.csv"

TRAIN_FILE_NAME:str="train.csv"

TEST_FILE_NAME:str="test.csv"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "Cardeo_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

######
# code for constant file for datavalidation
"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")