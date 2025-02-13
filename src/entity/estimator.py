# code for estimaytor

import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import Cardeo_risk_Exception
from src.logger import logging

#class TargetValueMapping:
#   def __init__(self):
#       self.Non_defaulter:int = 0
#       self.defaulter:int = 1
#   def _asdict(self):
#       return self.__dict__
#   def reverse_mapping(self):
#       mapping_response = self._asdict()
#      return dict(zip(mapping_response.values(),mapping_response.keys()))




# class CardeovasularModel:
#     def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
#         """
#         :param preprocessing_object: Input Object of preprocesser
#         :param trained_model_object: Input Object of trained model
#         """
#         self.preprocessing_object = preprocessing_object
#         self.trained_model_object = trained_model_object

#     def predict(self, dataframe: DataFrame) -> DataFrame:
#         """
#         Function accepts raw inputs and then transformed raw input using preprocessing_object
#         which guarantees that the inputs are in the same format as the training data
#         At last it performs prediction on transformed features
#         """
#         logging.info("Entered predict method of UTruckModel class")

#         try:
#             logging.info("Using the trained model to get predictions")
            

#             dataframe = pd.to_numeric(dataframe, errors='coerce')  # Convert non-numeric to NaN safely
            
#             if np.isnan(dataframe).any():
#                 logging.info("NaN values detected")
                
#             # Ensure the input is a Pandas Series or 1D NumPy array
#             if isinstance(data, dict):
#                 data = pd.Series(data)  # Convert dictionary to Pandas Series

#             elif isinstance(data, pd.DataFrame):
#                 data = data.squeeze()  # Convert single-column DataFrame to Series

#             elif isinstance(data, list) or isinstance(data, tuple):
#                 data = np.array(data)  # Convert list/tuple to NumPy array

#             else:
#                 raise ValueError(f"Unexpected input type: {type(data)}. Expected list, tuple, 1D array, or Series.")

#             print(f"Processed Data Type: {type(data)}")

#             transformed_feature = self.preprocessing_object.transform(dataframe)
#             ###################################################################
#             #transformed_feature = transformed_feature.dropna()
            

#             imputer = SimpleImputer(strategy='mean')  # You can use 'median' or 'most_frequent' based on your data
#             transformed_feature = imputer.fit_transform(transformed_feature)

#             ########################################################
#             import numpy as np

# # Check if there are any NaN values in the transformed_feature
#             # if np.isnan(transformed_feature).any():
#             #   raise ValueError("Input data contains NaN values. Please handle missing values before prediction.")
#             ######################################

#             logging.info("Used the trained model to get predictions")
#             return self.trained_model_object.predict(transformed_feature)

#         except Exception as e:
#             raise Cardeo_risk_Exception(e, sys) from e

#     def __repr__(self):
#         return f"{type(self.trained_model_object).__name__}()"

#     def __str__(self):
#         return f"{type(self.trained_model_object).__name__}()"

class CardeovasularModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Preprocessor pipeline object
        :param trained_model_object: Trained ML model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        
        
        
        
    ##########################################################
    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Transforms input data using the preprocessor and makes predictions.
        """
        logging.info("Entered predict method of CardiovascularModel class")
        try:
            logging.info("Using the trained model to get predictions")
            
            # Step 1: Ensure input is a Pandas DataFrame
            if isinstance(dataframe, dict):
                dataframe = pd.DataFrame([dataframe])  # Convert dictionary to DataFrame (single row)
            elif isinstance(dataframe, list) or isinstance(dataframe, tuple):
                dataframe = pd.DataFrame([dataframe])  # Convert list/tuple to DataFrame (single row)
            elif isinstance(dataframe, pd.Series):
                dataframe = dataframe.to_frame().T  # Convert Series to DataFrame
            
            # Step 2: Validate that the DataFrame has the expected number of features
            expected_features = [
                'age', 'education', 'BPMeds', 'cigsPerDay', 'prevalentStroke', 
                'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 
                'BMI', 'heartRate', 'glucose', 'sex', 'is_smoking'
            ]
            
            if list(dataframe.columns) != expected_features:
                raise ValueError(f"Unexpected columns in input data. Expected: {expected_features}, but got: {list(dataframe.columns)}")
            
            logging.info(f"Input Data Shape: {dataframe.shape}")
            logging.info(f"Columns: {list(dataframe.columns)}")
            logging.info(f"Data Types Before Conversion: \n{dataframe.dtypes}")
            
            # Step 3: Convert categorical columns to numeric
            if 'sex' in dataframe.columns:
                dataframe['sex'] = dataframe['sex'].map({'Male': 1, 'Female': 0}).fillna(0)
            if 'is_smoking' in dataframe.columns:
                dataframe['is_smoking'] = dataframe['is_smoking'].map({'Yes': 1, 'No': 0}).fillna(0)
            
            # Step 4: Convert all values to numeric, coercing errors to NaN
            dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
            
            logging.info(f"Data Types After Conversion: \n{dataframe.dtypes}")
            
            # Step 5: Handle missing values
            imputer = SimpleImputer(strategy='mean')
            dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
            
            # Step 6: Apply the preprocessing pipeline
            transformed_feature = self.preprocessing_object.transform(dataframe)
            
            # Step 7: Check for NaN values after transformation
            if np.isnan(transformed_feature).any():
                raise ValueError("NaN values remain in input data after imputation. Please check preprocessing.")
            
            logging.info("Successfully processed the input data, making predictions.")
            
            # Step 8: Make predictions using the trained model
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise Cardeo_risk_Exception(e, sys) from e
    ######################################################

    # def predict(self, dataframe: DataFrame) -> np.ndarray:
    #     """
    #     Transforms input data using the preprocessor and makes predictions.
    #     """
    #     logging.info("Entered predict method of CardiovascularModel class")

    #     try:
    #         logging.info("Using the trained model to get predictions")

    #         # Ensure input is a Pandas DataFrame
    #         if isinstance(dataframe, dict):
    #             dataframe = pd.DataFrame([dataframe])  # Convert dictionary to DataFrame (single row)

    #         elif isinstance(dataframe, list) or isinstance(dataframe, tuple):
    #             dataframe = pd.DataFrame([dataframe])  # Convert list/tuple to DataFrame (single row)

    #         elif isinstance(dataframe, pd.Series):
    #             dataframe = dataframe.to_frame().T  # Convert Series to DataFrame

    #         # Validate that the DataFrame has the expected number of features
    #         expected_features = [
    #             'age', 'education', 'BPMeds', 'cigsPerDay', 'prevalentStroke', 
    #             'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 
    #             'BMI', 'heartRate', 'glucose', 'sex', 'is_smoking'
    #         ]
            
    #         if list(dataframe.columns) != expected_features:
    #             raise ValueError(f"Unexpected columns in input data. Expected: {expected_features}, but got: {list(dataframe.columns)}")

    #         logging.info(f"Input Data Shape: {dataframe.shape}")
    #         logging.info(f"Columns: {list(dataframe.columns)}")
    #         logging.info(f"Data Types Before Conversion: \n{dataframe.dtypes}")

    #         # # Convert categorical columns to numeric
    #         # if 'sex' in dataframe.columns:
    #         #     dataframe['sex'] = dataframe['sex'].map({'Male': 1, 'Female': 0}).fillna(0)

    #         # if 'is_smoking' in dataframe.columns:
    #         #     dataframe['is_smoking'] = dataframe['is_smoking'].map({'Yes': 1, 'No': 0}).fillna(0)

    #         # Convert all values to numeric, coercing errors to NaN
    #         dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

    #         logging.info(f"Data Types After Conversion: \n{dataframe.dtypes}")

    #         # Handle missing values
    #         imputer = SimpleImputer(strategy='mean')
    #         transformed_feature = self.preprocessing_object.transform(dataframe)
    #         transformed_feature = imputer.fit_transform(transformed_feature)

    #         # Check for NaN values after transformation
    #         if np.isnan(transformed_feature).any():
    #             raise ValueError("NaN values remain in input data after imputation. Please check preprocessing.")

    #         logging.info("Successfully processed the input data, making predictions.")
    #         return self.trained_model_object.predict(transformed_feature)

    #     except Exception as e:
    #         logging.error(f"Prediction error: {str(e)}")
    #         raise Cardeo_risk_Exception(e, sys) from e

    # def __repr__(self):
    #     return f"{type(self.trained_model_object).__name__}()"

    # def __str__(self):
    #     return f"{type(self.trained_model_object).__name__}()"

    
