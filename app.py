# # ## app.py ## building the fast api

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import Response
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from starlette.responses import HTMLResponse, RedirectResponse
# from uvicorn import run as app_run

# from typing import Optional

# from src.constants import APP_HOST, APP_PORT
# from src.pipeline.prediction_pipeline import CardeoRiskData, CardeoRiskClassifier
# from src.pipeline.train_pipeline import TrainPipeline

# # app = FastAPI()

# # app.mount("/static", StaticFiles(directory="static"), name="static")

# # templates = Jinja2Templates(directory='templates')

# # origins = ["*"]

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # class DataForm:
# #     def __init__(self, request: Request):
# #         self.request: Request = request
# #         self.age: Optional[str] = None
# #         self.education: Optional[str] = None
# #         self.BPMeds:  Optional[str] = None
# #         self.cigsPerDay: Optional[str] = None
# #         self.prevalentStroke: Optional[str] = None
# #         self.prevalentHyp: Optional[str] = None
# #         self.diabetes: Optional[str] = None
# #         self.totChol: Optional[str] = None
# #         self.sysBP: Optional[str] = None
# #         self.diaBP: Optional[str] = None
# #         self.BMI: Optional[str] = None
# #         self.heartRate: Optional[str] = None
# #         self.glucose: Optional[str] = None
# #         self.sex: Optional[str] = None
# #         self.is_smoking: Optional[str] = None
  

# #     async def get_CardeoRisk_data(self):
# #         form = await self.request.form()
# #         self.age = form.get("age")
# #         self.education = form.get("education")
# #         self.BPMeds = form.get("BPMeds")
# #         self.cigsPerDay = form.get("cigsPerDay")
# #         self.prevalentStroke = form.get("prevalentStroke")
# #         self.prevalentHyp = form.get("prevalentHyp")
# #         self.diabetes = form.get("diabetes")
# #         self.totChol = form.get("totChol")
# #         self.sysBP = form.get("sysBP")
# #         self.diaBP = form.get("diaBP")
# #         self.BMI = form.get("BMI")
# #         self.heartRate = form.get("heartRate")
# #         self.glucose = form.get("glucose")
# #         self.sex = form.get("sex")
# #         self.is_smoking = form.get("is_smoking")
       

# # @app.get("/", tags=["authentication"])
# # async def index(request: Request):

# #     return templates.TemplateResponse(
# #             "CardeoRisk.html",{"request": request, "context": "Rendering"})


# # @app.get("/train")
# # async def trainRouteClient():
# #     try:
# #         train_pipeline = TrainPipeline()

# #         train_pipeline.run_pipeline()

# #         return Response("Training successful !!")

# #     except Exception as e:
# #         return Response(f"Error Occurred! {e}")


# # @app.post("/")
# # async def predictRouteClient(request: Request):
# #     try:
# #         form = DataForm(request)
# #         await form.get_CardeoRisk_data()

# #         CardeoRisk_data = CardeoRiskData(
# #                                 age= form.age,
# #                                 education = form.education,
# #                                 BPMeds = form.BPMeds,
# #                                 cigsPerDay = form.cigsPerDay,
# #                                 prevalentStroke= form.prevalentStroke,
# #                                 prevalentHyp= form.prevalentHyp,
# #                                 diabetes = form.diabetes,
# #                                 totChol= form.totChol,
# #                                 sysBP= form.sysBP,
# #                                 diaBP= form.diaBP,
# #                                 BMI= form.BMI,
# #                                 heartRate= form.heartRate,
# #                                 glucose= form.glucose,
# #                                 sex= form.sex,
# #                                 is_smoking= form.is_smoking,
                               
# #                                 )

# #         CardeoRisk_df = CardeoRisk_data.get_CardeoRisk_input_data_frame()

# #         model_predictor = CardeoRiskClassifier()

# #         value = model_predictor.predict(dataframe=CardeoRisk_df)[0]

# #         status = None
# #         if value == 1:
# #             status = "Heart_in Risk"
# #         else:
# #             status = "No Heart Risk"

# #         return templates.TemplateResponse(
# #             "CardeoRisk.html",
# #             {"request": request, "context": status},
# #         )

# #     except Exception as e:
# #         return {"status": False, "error": f"{e}"}


# # if __name__ == "__main__":
# #     app_run(app, host=APP_HOST, port=APP_PORT)



# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from starlette.responses import HTMLResponse
# from uvicorn import run as app_run

# from typing import Optional

# from src.constants import APP_HOST, APP_PORT
# from src.pipeline.prediction_pipeline import CardeoRiskData, CardeoRiskClassifier
# from src.pipeline.train_pipeline import TrainPipeline

# app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")

# templates = Jinja2Templates(directory='templates')

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class DataForm:
#     def __init__(self, request: Request):
#         self.request: Request = request
#         self.age: Optional[str] = None
#         self.education: Optional[str] = None
#         self.BPMeds:  Optional[str] = None
#         self.cigsPerDay: Optional[str] = None
#         self.prevalentStroke: Optional[str] = None
#         self.prevalentHyp: Optional[str] = None
#         self.diabetes: Optional[str] = None
#         self.totChol: Optional[str] = None
#         self.sysBP: Optional[str] = None
#         self.diaBP: Optional[str] = None
#         self.BMI: Optional[str] = None
#         self.heartRate: Optional[str] = None
#         self.glucose: Optional[str] = None
#         self.sex: Optional[str] = None
#         self.is_smoking: Optional[str] = None

#     async def get_CardeoRisk_data(self):
#         form = await self.request.form()
#         self.age = form.get("age")
#         self.education = form.get("education")
#         self.BPMeds = form.get("BPMeds")
#         self.cigsPerDay = form.get("cigsPerDay")
#         self.prevalentStroke = form.get("prevalentStroke")
#         self.prevalentHyp = form.get("prevalentHyp")
#         self.diabetes = form.get("diabetes")
#         self.totChol = form.get("totChol")
#         self.sysBP = form.get("sysBP")
#         self.diaBP = form.get("diaBP")
#         self.BMI = form.get("BMI")
#         self.heartRate = form.get("heartRate")
#         self.glucose = form.get("glucose")
#         self.sex = form.get("sex")
#         self.is_smoking = form.get("is_smoking")

# @app.get("/", tags=["authentication"])
# async def index(request: Request):
#     return templates.TemplateResponse(
#             "CardeoRisk.html",{"request": request, "context": "Rendering"})

# @app.get("/train")
# async def trainRouteClient():
#     try:
#         train_pipeline = TrainPipeline()
#         train_pipeline.run_pipeline()
#         return Response("Training successful !!")
#     except Exception as e:
#         return Response(f"Error Occurred! {e}")

# @app.post("/predict")
# async def predictRouteClient(request: Request):
#     try:
#         form = DataForm(request)
#         await form.get_CardeoRisk_data()

#         CardeoRisk_data = CardeoRiskData(
#             age=form.age,
#             education=form.education,
#             BPMeds=form.BPMeds,
#             cigsPerDay=form.cigsPerDay,
#             prevalentStroke=form.prevalentStroke,
#             prevalentHyp=form.prevalentHyp,
#             diabetes=form.diabetes,
#             totChol=form.totChol,
#             sysBP=form.sysBP,
#             diaBP=form.diaBP,
#             BMI=form.BMI,
#             heartRate=form.heartRate,
#             glucose=form.glucose,
#             sex=form.sex,
#             is_smoking=form.is_smoking,
#         )

#         CardeoRisk_df = CardeoRisk_data.get_CardeoRisk_input_data_frame()

#         model_predictor = CardeoRiskClassifier()
#         value = model_predictor.predict(dataframe=CardeoRisk_df)[0]

#         status = "Heart at Risk" if value == 1 else "No Heart Risk"

#         return JSONResponse(content={"status": status})

#     except Exception as e:
#         return JSONResponse(content={"status": "Error", "error": str(e)})

# if __name__ == "__main__":
#     app_run(app, host=APP_HOST, port=APP_PORT)




import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from uvicorn import run as app_run

from src.constants import APP_HOST, APP_PORT
from typing import Optional
from src.pipeline.prediction_pipeline import CardeoRiskData, CardeoRiskClassifier
from src.pipeline.train_pipeline import TrainPipeline


# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Model for Input Validation
class CardioRiskRequest(BaseModel):
    age: float
    education: int
    BPMeds: Optional[int] = 0
    cigsPerDay: Optional[float] = 0.0
    prevalentStroke: Optional[int] = 0
    prevalentHyp: Optional[int] = 0
    diabetes: Optional[int] = 0
    totChol: Optional[float] = None
    sysBP: Optional[float] = None
    diaBP: Optional[float] = None
    BMI: Optional[float] = None
    heartRate: Optional[float] = None
    glucose: Optional[float] = None
    sex: Optional[int] = None
    is_smoking: Optional[int] = 0

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse("CardeoRisk.html", {"request": request, "context": "Rendering"})

@app.get("/train")
async def trainRouteClient():
    try:
        logger.info("Training pipeline started.")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logger.info("Training pipeline completed successfully.")
        return JSONResponse(content={"status": "Training successful!"})
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return JSONResponse(content={"status": "Error", "error": str(e)}, status_code=500)
    
    
@app.post("/predict")
async def predictRouteClient(data: CardioRiskRequest):
    try:
        logger.info(f"Received prediction request: {data}")
        
        # Convert request data to DataFrame
        cardio_risk_data = CardeoRiskData(**data.dict())
        cardio_risk_df = cardio_risk_data.get_CardeoRisk_input_data_frame()
        
        ##########################
        # Preprocess the input data
        # Ensure all values are numeric and handle missing values
        cardio_risk_df = cardio_risk_df.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
        
        # Handle missing values using an imputer
        imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean of each column
        cardio_risk_df = pd.DataFrame(imputer.fit_transform(cardio_risk_df), columns=cardio_risk_df.columns)
        
        # Ensure the input data has the correct number of features
        expected_features = ['age', 'education', 'BPMeds', 'cigsPerDay', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'sex', 'is_smoking']
        missing_features = set(expected_features) - set(cardio_risk_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder columns to match the expected order
        cardio_risk_df = cardio_risk_df[expected_features]
        
        ##########################
        # Make prediction
        model_predictor = CardeoRiskClassifier()
        prediction = model_predictor.predict(dataframe=cardio_risk_df)[0]
        
        # Determine status
        status = "Heart at Risk" if prediction == 1 else "No Heart Risk"
        logger.info(f"Prediction result: {status}")
        return JSONResponse(content={"status": status})
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(content={"status": "Error", "error": str(e)}, status_code=500)

# @app.post("/predict")
# async def predictRouteClient(data: CardioRiskRequest):
#     try:
#         logger.info(f"Received prediction request: {data}")
        
#         # Convert request data to DataFrame
#         cardio_risk_data = CardeoRiskData(**data.dict())
#         cardio_risk_df = cardio_risk_data.get_CardeoRisk_input_data_frame()
        
#         ##########################

#         # Make prediction
#         model_predictor = CardeoRiskClassifier()
#         prediction = model_predictor.predict(dataframe=cardio_risk_df)[0]

#         # Determine status
#         status = "Heart at Risk" if prediction == 1 else "No Heart Risk"
#         logger.info(f"Prediction result: {status}")

#         return JSONResponse(content={"status": status})
    
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         return JSONResponse(content={"status": "Error", "error": str(e)}, status_code=500)

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
