from fastapi import FastAPI

from pydantic import BaseModel

from prediction import predict
 
app = FastAPI(title="Diabetes Prediction API")
 
class DiabetesInput(BaseModel):

    Pregnancies: int

    Glucose: float

    BloodPressure: float

    SkinThickness: float

    Insulin: float

    BMI: float

    DiabetesPedigreeFunction: float

    Age: int
 
@app.get("/")

def home():

    return {"message": "Diabetes Prediction API is running."}
 
@app.post("/predict")

def predict_diabetes(input_data: DiabetesInput):

    ordered_data = {

        "Pregnancies": input_data.Pregnancies,

        "Glucose": input_data.Glucose,

        "BloodPressure": input_data.BloodPressure,

        "SkinThickness": input_data.SkinThickness,

        "Insulin": input_data.Insulin,

        "BMI": input_data.BMI,

        "DiabetesPedigreeFunction": input_data.DiabetesPedigreeFunction,

        "Age": input_data.Age

    }

    result = predict(ordered_data)

    return result
 
