#entire code of Data_Transformation.py

import sys
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import Cardeo_risk_Exception
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
#from Demo_project.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise Cardeo_risk_Exception(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Cardeo_risk_Exception(e, sys)


    # @staticmethod
    # def replace_to_zero(df, columns):
    #     """
    #     Replace invalid values (-2, -1, 0) in specified columns with 0.
    #     """
    #     try:
    #         for col in columns:
    #             fil = (df[col] == -2) | (df[col] == -1) | (df[col] == 0)
    #             df.loc[fil, col] = 0
    #         return df
    #     except Exception as e:
    #         raise Cardeo_risk_Exception(e, sys)
    @staticmethod
    def remove_duplicates(df):
        """
        Remove duplicate rows from the DataFrame.
       """
        try:
            before_count = df.shape[0]
            df = df.drop_duplicates()
            after_count = df.shape[0]
            logging.info(f"Removed {before_count - after_count} duplicate rows.")
            logging.info(f"Removed  duplicate rows.")
            return df
        except Exception as e:
            raise Cardeo_risk_Exception(e, sys)
    @staticmethod
    def replace_null_with_median(df):
        """
        Replace null values in specified columns with the median.
        """
        try:
            columns = ['education', 'cigsPerDay', 'BPMeds']
            for column in columns:
                if column in df.columns:  # Check if the column exists in the DataFrame
                    df[column] = df[column].fillna(df[column].median(), inplace=False)
                else:
                    raise KeyError(f"Column '{column}' not found in the DataFrame.")
            logging.info("Missing values filled with median successfully.")
            return df
        except Exception as e:
            logging.error(f"Error occurred while replacing nulls with median: {e}")
            raise Cardeo_risk_Exception(e, sys)

    @staticmethod
    def replace_null_with_mean(df):
        """
        Replace null values in specified columns with the mean of those columns.
        """
        try:
            columns = ['totChol', 'BMI', 'heartRate']  # List of columns to process
            for col in columns:  # Iterate over the columns
                if col in df.columns:  # Check if the column exists in the DataFrame
                    df[col] = df[col].fillna(df[col].mean())  # Replace nulls with the mean
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")  # Raise error if column is missing
            logging.info("Missing values filled with mean successfully.")
            return df
        except Exception as e:
            logging.error(f"Error occurred while replacing nulls with mean: {e}")
            raise Cardeo_risk_Exception(e, sys)

    @staticmethod
    def impute_glucose_with_knn(df):
        """
        Impute missing values in the specified column using KNN.

        Parameters:
        - df: pandas DataFrame

        Returns:
        - Updated DataFrame with imputed values for the specified column.
        """
        try:
            # Check if the column exists in the DataFrame
            column_name = 'glucose'
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in DataFrame.")

            # Defining the KNN imputer
            imputer = KNNImputer()

            # Reshape the column and apply the imputer
            df[column_name] = imputer.fit_transform(df[[column_name]])  # Use double brackets for 2D input
            logging.info("Missing values in 'glucose' column imputed with KNN successfully.")
            return df
        except Exception as e:
            logging.error(f"Error occurred while imputing glucose with KNN: {e}")
            raise Cardeo_risk_Exception(e, sys)

    # @staticmethod
    # def process_cigs_per_day_column(df):
    #     try:
    #         df.rename(columns={'cigsPerDay': 'cigs_per_day'}, inplace=True)
    #         # Convert the column to numeric, coercing non-numeric values to NaN
    #         df['cigs_per_day'] = pd.to_numeric(df['cigs_per_day'], errors='coerce')

    #         # Replace NaN values with 0 or another appropriate default
    #         df['cigs_per_day'].fillna(0, inplace=True)

    #         # Ensure all values are integers
    #         df['cigs_per_day'] = df['cigs_per_day'].astype(int)

    #         # Apply your logic
    #         for i in range(len(df)):
    #             if 0 < df['cigs_per_day'][i] < 20:
    #                 df['cigs_per_day'][i] = 10
    #             elif 20 <= df['cigs_per_day'][i]:
    #                 df['cigs_per_day'][i] = 30

    #         return df

    #     except Exception as e:
    #         #logging.error(f"Error at index {i}, value: {df.loc[i, 'cigs_per_day']}")
    #         raise Cardeo_risk_Exception(e, sys)

    # def process_cigs_per_day_column(df):
    #     """
    #     Process the 'cigsPerDay' column:
    #     - Rename the column to 'cigs_per_day'.
    #     - Update the values based on consumption levels:
    #         - 0: 'No Consumption'
    #         - >0 and <20: 'Average Consumption'
    #         - >=20: 'High Consumption'
    #     """
    #     try:
    #         # Rename the column
    #         df.rename(columns={'cigsPerDay': 'cigs_per_day'}, inplace=True)
    #         df['cigs_per_day'] = df['cigs_per_day'].astype(str)

    #         # Update values in the renamed column
    #         # for i in range(len(df)):
    #         #     if df['cigs_per_day'][i] == 0:
    #         for i, row in df.iterrows():
    #             if row['cigs_per_day'] == '0':  # or whatever your condition is
    #                 df.loc[i, 'cigs_per_day'] = 'No Consumption'

    #                 #  df['cigs_per_day'][i] = 'No Consumption'
    #             elif 0 < df['cigs_per_day'][i] < 20:
    #                 df['cigs_per_day'][i] = 'Average Consumption'
    #             else:
    #                 df['cigs_per_day'][i] = 'High Consumption'

    #         return df
    #     except Exception as e:
    #         logging.error(f"Error at index {i}, value: {df.loc[i, 'cigs_per_day']}")
    #         raise Cardeo_risk_Exception(e, sys)





    @staticmethod
    def remove_outliers(df, columns):
        """
        Remove outliers using IQR method for specified columns.
        """
        try:
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            return df
        except Exception as e:
            raise Cardeo_risk_Exception(e, sys)

    
    
    @staticmethod
    def get_data_transformer_object(schema_config) -> Pipeline:
        """
        Creates and returns a data transformer object for the data.
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )
        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),  # Impute missing values for numerical columns
                ("scaler", StandardScaler())  # Scaling numerical features
            ])

            oh_transformer = OneHotEncoder()  # One-hot encoding for categorical features

            logging.info("Initialized StandardScaler, OneHotEncoder")

            # Get columns from schema configuration
            num_features = schema_config['num_features']
            oh_columns = schema_config['oh_columns']
            transform_columns = schema_config['transform_columns']

            logging.info("Initialize PowerTransformer")
            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            # Combining all transformations using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )
            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")

            return preprocessor
        except Exception as e:
            raise Cardeo_risk_Exception(e, sys) from e





    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline

        Output      :   data transformer steps are performed and preprocessor object is created
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                
                # Get the preprocessor object
                preprocessor = self.get_data_transformer_object(self._schema_config)
                logging.info("Got the preprocessor object")

                # Read train and test datasets
                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Replace null values using different methods
                logging.info("Replacing null values in train and test datasets")
                train_df = self.replace_null_with_median(train_df)
                test_df = self.replace_null_with_median(test_df)

                train_df = self.replace_null_with_mean(train_df)
                test_df = self.replace_null_with_mean(test_df)

                train_df = self.impute_glucose_with_knn(train_df)
                test_df = self.impute_glucose_with_knn(test_df)

                # Remove duplicates from datasets
                train_df = self.remove_duplicates(train_df)
                test_df = self.remove_duplicates(test_df)

                # Drop target column to create input features
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                # Drop unnecessary columns
                drop_cols = self._schema_config['drop_columns']
                logging.info("Dropping unnecessary columns from train and test datasets")
                input_feature_train_df = input_feature_train_df.drop(columns=drop_cols, errors='ignore')
                input_feature_test_df = input_feature_test_df.drop(columns=drop_cols, errors='ignore')

                # Transform features using the preprocessor
                logging.info("Applying preprocessing object on train and test features")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                # Apply SMOTEENN on training and testing data
                logging.info("Applying SMOTEENN on training and testing datasets")
                smt = SMOTEENN(sampling_strategy="minority")
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                # Combine input and target features into arrays
                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                # Save preprocessor object and transformed data
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
                logging.info("Saved the preprocessor object and transformed datasets")

                # Create and return the data transformation artifact
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            logging.error(f"Error occurred during data transformation: {e}")
            raise Cardeo_risk_Exception(e, sys) from e

#     def initiate_data_transformation(self) -> DataTransformationArtifact:
#         """
#         Initiates the data transformation component for the pipeline.
#         """
#         try:
#             if self.data_validation_artifact.validation_status:
#                 logging.info("Starting data transformation")
                
#                 # Use static method
#                 preprocessor = self.get_data_transformer_object(self._schema_config)
#                 logging.info("Got the preprocessor object")

#                 train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
#                 test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

#                 train_df = self.remove_duplicates(train_df)
#                 test_df = self.remove_duplicates(test_df)

#                 input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
#                 target_feature_train_df = train_df[TARGET_COLUMN]

#                 drop_cols = self._schema_config['drop_columns']

#                 logging.info("Dropping the columns in drop_cols of Training dataset")
#                 input_feature_train_df = input_feature_train_df.drop(columns=drop_cols, errors='ignore')

#                 input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
#                 target_feature_test_df = test_df[TARGET_COLUMN]

#                 input_feature_test_df = input_feature_test_df.drop(columns=drop_cols, errors='ignore')
# ##############################check
#                 #logging.info(f"Columns in input_feature_train_df: {input_feature_train_df.columns.tolist()}")
#                 #logging.info(f"Expected columns in schema: {self._schema_config}")
                
#                 #logging.info(f"Columns to be dropped: {drop_cols}")
#                 #logging.info(f"Remaining columns after dropping: {input_feature_train_df.columns.tolist()}")

# ################################################

#                 logging.info("Applying preprocessing object on training and testing datasets")
#                 input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
#                 input_feature_test_arr = preprocessor.transform(input_feature_test_df)

#                 logging.info("Applying SMOTEENN on Training and Testing datasets")
#                 smt = SMOTEENN(sampling_strategy="minority")
#                 input_feature_train_final, target_feature_train_final = smt.fit_resample(
#                     input_feature_train_arr, target_feature_train_df
#                 )
#                 input_feature_test_final, target_feature_test_final = smt.fit_resample(
#                     input_feature_test_arr, target_feature_test_df
#                 )

#                 train_arr = np.c_[
#                     input_feature_train_final, np.array(target_feature_train_final)
#                 ]
#                 test_arr = np.c_[
#                     input_feature_test_final, np.array(target_feature_test_final)
#                 ]

#                 save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
#                 save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
#                 save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

#                 logging.info("Saved the preprocessor object")
#                 logging.info(
#                     "Exited initiate_data_transformation method of DataTransformation class"
#                 )

#                 data_transformation_artifact = DataTransformationArtifact(
#                     transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                     transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
#                     transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
#                 )
#                 return data_transformation_artifact
#             else:
#                 raise Exception(self.data_validation_artifact.message)
#         except Exception as e:
#             raise Cardeo_risk_Exception(e, sys) from e
    
    
    


    