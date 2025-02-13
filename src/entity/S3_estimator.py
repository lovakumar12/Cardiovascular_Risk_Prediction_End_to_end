#code for s3_estimator

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import Cardeo_risk_Exception
from src.entity.estimator import CardeovasularModel
import sys
from pandas import DataFrame
from src.logger import logging


class CardeoRiskEstimator:
    """
    This class is used to save and retrieve us_visas model in s3 bucket and to do prediction
    """

    def __init__(self,bucket_name,model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model:CardeovasularModel=None


    def is_model_present(self,model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except Cardeo_risk_Exception as e:
            print(e)
            return False

    def load_model(self,)->CardeovasularModel:
        """
        Load the model from the model_path
        :return:
        """

        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)

    def save_model(self,from_file,remove:bool=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        except Exception as e:
            raise Cardeo_risk_Exception(e, sys)

    
    ###########################################################
    def predict(self, dataframe: DataFrame):
        """
        Predict using the loaded model
        :param dataframe: Input DataFrame for prediction
        :return: Prediction result
        """
        try:
            # Step 1: Load the model if not already loaded
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
                logging.info("Model Loaded Successfully")
            
            # Step 2: Log the input data for debugging
            logging.info(f"Data Type Before Validation: {type(dataframe)}")
            logging.info(f"Data Before Validation:\n{dataframe.head()}")
            
            # Step 3: Ensure all columns are numeric
            if not dataframe.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
                raise ValueError("Input DataFrame contains non-numeric columns!")
            
            # Step 4: Handle missing values (replace NaN with 0 or another strategy)
            if dataframe.isnull().values.any():
                logging.warning("NaN values found in input DataFrame. Filling with 0.")
                dataframe.fillna(0, inplace=True)
            
            # Step 5: Log the processed data for debugging
            logging.info(f"Processed Data:\n{dataframe.head()}")
            
            # Step 6: Make predictions using the loaded model
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise Cardeo_risk_Exception(e, sys)
    #############################################################
    # def predict(self,dataframe:DataFrame):
    #     """
    #     :param dataframe:
    #     :return:
    #     """
    #     try:
    #         if self.loaded_model is None:
    #             self.loaded_model = self.load_model()
    #             logging.info(f"Model Loaded Successfully")
    #             logging.info(f"Data Type Before NaN Check: {type(dataframe)}")
    #             logging.info(f"Data Before NaN Check: {dataframe}")
                

    #         return self.loaded_model.predict(dataframe=dataframe)
    #     except Exception as e:
    #         raise Cardeo_risk_Exception(e, sys)