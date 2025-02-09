# code of prediction_pipeline


import os
import sys

import numpy as np
import pandas as pd
from src.entity.config_entity import CardeoRiskPredictorConfig
from src.entity.S3_estimator import CardeoRiskEstimator
from src.exception import Cardeo_risk_Exception
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from pandas import DataFrame


class CardeoRiskData:
    def __init__(self,
                age,
                education,
                cigsPerDay,
                BPMeds,
                prevalentStroke,
                prevalentHyp,
                diabetes,
                totChol,
                sysBP,
                diaBP,
                BMI,
                heartRate ,
                glucose,
                sex,
                is_smoking,
                
                ):
        """
        CardeoRisk Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.age = age
            self.education = education
            self.cigsPerDay = cigsPerDay
            self.BPMeds = BPMeds
            self.prevalentStroke = prevalentStroke
            self.prevalentHyp = prevalentHyp
            self.diabetes = diabetes
            self.totChol = totChol
            self.sysBP = sysBP
            self.diaBP = diaBP
            self.BMI = BMI
            self.heartRate = heartRate
            self.glucose = glucose
            self.sex = sex
            self.is_smoking = is_smoking
            


        except Exception as e:
            raise Cardeo_risk_Exception(e, sys) from e
        
        
    def get_CardeoRisk_input_data_frame(self) -> DataFrame:
        try:
            CardeoRisk_input_dict = self.get_CardeoRisk_data_as_dict()
            df = DataFrame(CardeoRisk_input_dict)

            # Convert all columns to numeric, forcing errors if any non-numeric values exist
            df = df.apply(pd.to_numeric, errors='coerce')

            logging.info(f"Processed input DataFrame:\n{df.head()}")

            return df
        except Exception as e:
            raise Cardeo_risk_Exception(e, sys)


    # def get_CardeoRisk_input_data_frame(self)-> DataFrame:
    #     """
    #     This function returns a DataFrame from CardeoRiskData class input
    #     """
    #     try:

    #         CardeoRisk_input_dict = self.get_CardeoRisk_data_as_dict()
    #         return DataFrame(CardeoRisk_input_dict)
            
    #         # Convert all columns to numeric, forcing errors if any non-numeric values exist
    #         df = df.apply(pd.to_numeric, errors='coerce')
            
            
    #     except Exception as e:
    #         raise Cardeo_risk_Exception(e, sys) from e


    def get_CardeoRisk_data_as_dict(self):
        """
        This function returns a dictionary from CardeoRisk Data class input
        """
        logging.info("Entered get_CardeoRisk_data_as_dict method as CardeoRiskData class")

        try:
            input_data = {
                "age": [self.age],
                "education": [self.education],
                "BPMeds": [self.BPMeds],
                "cigsPerDay": [self.cigsPerDay],
                "prevalentStroke": [self.prevalentStroke],
                "prevalentHyp": [self.prevalentHyp],
                "diabetes": [self.diabetes],
                "totChol": [self.totChol],
                "sysBP": [self.sysBP],
                "diaBP": [self.diaBP],
                "BMI": [self.BMI],
                "heartRate": [self.heartRate],
                "glucose": [self.glucose],
                "sex": [self.sex],
                "is_smoking": [self.is_smoking],
               

            }

            logging.info("Created CardeoRisk data dict")

            logging.info("Exited get_CardeoRisk_data_as_dict method as CardeoRiskData class")

            return input_data

        except Exception as e:
            raise Cardeo_risk_Exception(e, sys) from e

class CardeoRiskClassifier:
    def __init__(self,prediction_pipeline_config: CardeoRiskPredictorConfig = CardeoRiskPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise Cardeo_risk_Exception (e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of CardeoRiskClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of CardeoRiskClassifier class")
            
            # Print data types to check for invalid types
            logging.info(f"Data types of input dataframe: {dataframe.dtypes}")
            logging.info(f"First few rows of input dataframe:\n{dataframe.head()}")
            model = CardeoRiskEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)

            return result

        except Exception as e:
            raise Cardeo_risk_Exception(e, sys)
    
    