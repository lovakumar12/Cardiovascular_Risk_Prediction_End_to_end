# from src.constants import DATABASE_NAME ,COLLECTION_NAME,MONGODB_URL_KEY
# print(DATABASE_NAME)
# print(COLLECTION_NAME)

#from src.logger import * (got some error)
#from src.utils import *  (working fine)
#from src.exception import *  #(working fine)
#from src.configuration import *  ##(working fine)


##########to check the configuration

# from src.configuration.mongo_db_connection import MongoDBClient

# def main():
#     try:
#         # Initialize MongoDB client
#         mongo_client = MongoDBClient()  # Default uses DATABASE_NAME

#         # Print connection details for confirmation
#         print(f"Connected to MongoDB database: {mongo_client.database_name}")
#         print(f"Client: {mongo_client.client}")

#         # Accessing a specific collection (example)
#         collection_name = "Cardeo_data"  # Replace with your collection name
#         collection = mongo_client.database[collection_name]
#         print(f"Connected to collection: {collection_name}")

#         # Fetching and displaying documents from the collection
#         documents = collection.find()
#         print("Documents in the collection:")
#         for doc in documents:
#             print(doc)

#     except Exception as e:
#         print(f"Error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()

###################to check the data access

# from src.configuration.mongo_db_connection import MongoDBClient
# from src.constants import DATABASE_NAME
# from src.data_access.cardeo_data import  cardeovascular_risk_Data
# from src.exception import Cardeovascular_risk_exception

# def main():
#     try:
#         # Create an instance of Credit_Fraud_Data
#         fraud_data = cardeovascular_risk_Data()

#         # Specify the collection name
#         collection_name = "Cardeo_data"  # Replace with your actual collection name

#         # Call the export_collection_as_dataframe method
#         df = fraud_data.export_collection_as_dataframe(collection_name=collection_name)

#         # Print the dataframe for verification
#         print("Exported DataFrame:")
#         print(df.head(10))  # Print first few rows

#     except Cardeovascular_risk_exception as e:
#         print(f"Custom Exception Occurred: {str(e)}")
#     except Exception as e:
#         print(f"General Exception Occurred: {str(e)}")

# if __name__ == "__main__":
#     main()


################### checking the entity

# this is the code we have to use in demo.py
# from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
# from src.entity.artifact_entity import DataIngestionArtifact

# def main():
#     try:
#         # Initialize TrainingPipelineConfig
#         training_pipeline_config = TrainingPipelineConfig()
#         print("Training Pipeline Config:")
#         print(f"Pipeline Name: {training_pipeline_config.pipeline_name}")
#         print(f"Artifact Directory: {training_pipeline_config.artifact_dir}")
#         print(f"Timestamp: {training_pipeline_config.timestamp}")

#         # Initialize DataIngestionConfig
#         data_ingestion_config = DataIngestionConfig()
#         print("\nData Ingestion Config:")
#         print(f"Data Ingestion Directory: {data_ingestion_config.data_ingestion_dir}")
#         print(f"Feature Store File Path: {data_ingestion_config.feature_store_file_path}")
#         print(f"Training File Path: {data_ingestion_config.training_file_path}")
#         print(f"Testing File Path: {data_ingestion_config.testing_file_path}")
#         print(f"Train-Test Split Ratio: {data_ingestion_config.train_test_split_ratio}")
#         print(f"Collection Name: {data_ingestion_config.collection_name}")

#         # Initialize DataIngestionArtifact with sample file paths
#         sample_trained_path = data_ingestion_config.training_file_path
#         sample_test_path = data_ingestion_config.testing_file_path
#         data_ingestion_artifact = DataIngestionArtifact(
#             trained_file_path=sample_trained_path,
#             test_file_path=sample_test_path,
#         )

#         print("\nData Ingestion Artifact:")
#         print(f"Trained File Path: {data_ingestion_artifact.trained_file_path}")
#         print(f"Test File Path: {data_ingestion_artifact.test_file_path}")

#     except Exception as e:
#         print(f"Error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()


################# checking the data_ingension.py

# testing of code of data_ingestin using  demo.py
# import sys
# from src.components.data_ingenstion import DataIngestion
# from src.entity.config_entity import DataIngestionConfig
# from src.logger import logging
# from src.exception import Cardeovascular_risk_exception

# def main():
#     try:
#         # Initialize DataIngestionConfig
#         data_ingestion_config = DataIngestionConfig()

#         # Initialize DataIngestion
#         data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

#         # Start the data ingestion process
#         data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

#         # Log the results
#         logging.info(f"Data ingestion completed successfully.")
#         logging.info(f"Training file path: {data_ingestion_artifact.trained_file_path}")
#         logging.info(f"Testing file path: {data_ingestion_artifact.test_file_path}")

#         # Print dataset shapes for confirmation
#         import pandas as pd
#         train_data = pd.read_csv(data_ingestion_artifact.trained_file_path)
#         test_data = pd.read_csv(data_ingestion_artifact.test_file_path)
#         print(f"Training dataset shape: {train_data.shape}")
#         print(f"Testing dataset shape: {test_data.shape}")
#     except Exception as e:
#         logging.error(f"Error occurred during data ingestion: {e}")
#         print(f"Error occurred during data ingestion: {e}")

# if __name__ == "__main__":
#     main()

#####################checking the data_validation.py
#code in demo.py

# from src.pipeline.train_pipeline import TrainPipeline
# pipeline=TrainPipeline()
# pipeline.run_pipeline()

# Example test
from src.pipeline.prediction_pipeline import CardeoRiskData, CardeoRiskClassifier
from src.entity.S3_estimator import CardeoRiskEstimator
import pandas as pd

# Sample input data
input_data = CardeoRiskData(
    age=45,
    education="high school",
    cigsPerDay=10,
    BPMeds=0,
    prevalentStroke=0,
    prevalentHyp=1,
    diabetes=0,
    totChol=200,
    sysBP=120,
    diaBP=80,
    BMI=25,
    heartRate=70,
    glucose=100,
    sex="Male",
    is_smoking="Yes"
)

# Get input DataFrame
input_df = input_data.get_CardeoRisk_input_data_frame()

# Initialize estimator and make prediction
estimator = CardeoRiskEstimator(bucket_name="your_bucket_name", model_path="your_model_path")
prediction = estimator.predict(input_df)
print("Prediction:", prediction)