from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles     
from fastapi.responses import FileResponse       
from pydantic import BaseModel
import pandas as pd
import joblib
import tensorflow as tf
import os

app = FastAPI(title="Delivery Time Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/delivery_model.keras")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "../../models/preprocessor.pkl")
STATIC_PATH = os.path.join(BASE_DIR, "static")


model = None
preprocessor = None

class DeliveryInput(BaseModel):
    distance_km: float
    package_weight_kg: float
    delivery_rating: float
    delivery_cost: float
    vehicle_type: str
    weather_condition: str
    delivery_mode: str
    region: str
    package_type: str = "electronics"
    delivery_partner: str = "dhl"

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
   
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Modelo y Preprocesador cargados correctamente.")
    else:
        print(f"ERROR: No se encuentran los archivos en:\n{MODEL_PATH}\n{PREPROCESSOR_PATH}")


@app.get("/")
def read_index():
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

@app.post("/predict")
def predict_delivery_time(data: DeliveryInput):
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    
    try:
        input_df = pd.DataFrame([data.dict()])
        
        input_processed = preprocessor.transform(input_df)
        
        prediction = model.predict(input_processed)
        predicted_hours = float(prediction[0][0])
        
        return {
            "predicted_hours": round(predicted_hours, 2),
            "predicted_minutes": round(predicted_hours * 60, 0),
            "status": "success"
        }
    except Exception as e:
        print(f"Error en predicción: {e}")
        raise HTTPException(status_code=400, detail=str(e))