import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Global State for Model & Scalers (Cached via Lifespan)
# ---------------------------------------------------------
ml_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On Startup: Load the heavy CNN model, the Data Scalers, and 
    calculate the Dataset Mean/Std to be kept in memory.
    """
    print("Booting AI Engine...")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    transformer_path = os.path.join(data_dir, "best_transformer_model.keras")
    cnn_path = os.path.join(data_dir, "best_cnn_model.keras")
    feature_scaler_path = os.path.join(data_dir, "feature_scaler.save")
    target_scaler_path = os.path.join(data_dir, "target_scaler.save")
    ytest_path = os.path.join(data_dir, "y_test.npy")

    if not all(os.path.exists(p) for p in [feature_scaler_path, target_scaler_path, ytest_path]):
        raise RuntimeError("Critical ML assets are missing. Please complete layers 1-6 first.")

    # Load Transformer (primary) or CNN (fallback)
    model_errors = []
    for model_name, model_path in [("Transformer", transformer_path), ("CNN", cnn_path)]:
        if not os.path.exists(model_path):
            continue
        try:
            ml_components["model"] = load_model(model_path)
            ml_components["model_name"] = model_name
            break
        except Exception as model_error:
            model_errors.append(f"{model_name}: {model_error}")

    if "model" not in ml_components:
        # Surface the real load failures - swallowing them reports a corrupt
        # checkpoint as "no model found", which sends you looking in the wrong place.
        detail = "; ".join(model_errors) if model_errors else "no checkpoint files present"
        raise RuntimeError(f"No trained model found. Run Layer 5/5b first. Details: {detail}")
    ml_components["feature_scaler"] = joblib.load(feature_scaler_path)
    ml_components["target_scaler"] = joblib.load(target_scaler_path)

    # Reference distribution for the confidence check.
    # This is derived from the GROUND-TRUTH targets, not from the model's own
    # predictions: predictions are smoother than reality, so using them understates
    # the spread and makes the 2-sigma / 3-sigma anomaly thresholds fire too eagerly.
    ytest_path = os.path.join(data_dir, "y_test.npy")
    y_scaled = np.load(ytest_path).reshape(-1, 1)
    y_true = ml_components["target_scaler"].inverse_transform(y_scaled).flatten()
    ml_components["mean_emission"] = float(np.mean(y_true))
    ml_components["std_emission"] = float(np.std(y_true))

    print(f"AI Engine Loaded and Ready! (Model: {ml_components['model_name']})")
    
    yield
    
    # On Shutdown: Clean up memory
    print("Shutting down AI Engine...")
    ml_components.clear()

# ---------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Carbon Emission Forecasting API",
    description="REST API serving the Carbon Emission Transformer/CNN model.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow browser clients from any origin to call this read-only inference API.
# allow_credentials MUST stay False while allow_origins is "*": browsers reject that
# combination outright, so the previous config silently blocked every real CORS request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Pydantic Schemas (Strict Input Validation)
# ---------------------------------------------------------
class EmissionRequest(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50.0, le=60.0)
    humidity: float = Field(..., description="Humidity percentage", ge=0.0, le=100.0)
    wind_speed: float = Field(..., description="Wind speed in km/h", ge=0.0)
    industry_level: int = Field(..., description="Industry categorical level", ge=1, le=5)
    vehicle_count: int = Field(..., description="Daily vehicle traffic", ge=0)
    energy_usage: float = Field(..., description="Total Energy used in MWh", ge=0.0)

class EmissionResponse(BaseModel):
    predicted_emission: float
    status_level: str
    confidence: str

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """
    Send the bare URL to the interactive docs.

    Without this route http://localhost:8000/ returned {"detail":"Not Found"},
    which reads like a broken server even though the API was healthy.
    """
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """DevOps ping to verify the server and model are online."""
    return {"status": "online", "model_loaded": "model" in ml_components, "model_name": ml_components.get("model_name", "none")}

@app.post("/predict", response_model=EmissionResponse)
def predict_emission(request: EmissionRequest):
    """Core Inference Pipeline endpoint."""
    if "model" not in ml_components:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # 1. Transform JSON into Raw Feature Array
        raw_features = np.array([[
            request.temperature,
            request.humidity,
            request.wind_speed,
            request.industry_level,
            request.vehicle_count,
            request.energy_usage
        ]])

        # 2. Preprocess & Scale (Using pre-fit scaler)
        scaler_X = ml_components["feature_scaler"]
        scaled_features = scaler_X.transform(raw_features)

        # 3. Simulate Sequential Window
        # The model expects shape (samples, 7_time_steps, 6_features)
        time_steps = 7
        sequence = np.tile(scaled_features, (time_steps, 1))
        sequence = np.expand_dims(sequence, axis=0) # Shape: (1, 7, 6)
        
        # 4. Neural Network Inference
        model = ml_components["model"]
        pred_scaled = model.predict(sequence, verbose=0)
        
        # 5. Post-Process (Inverse Transform)
        scaler_y = ml_components["target_scaler"]
        pred_true = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
        
        # 6. Safety Status Logic
        if pred_true < 2000:
            status = "Safe"
        elif pred_true <= 6000:
            status = "Warning"
        else:
            status = "Danger"
            
        # 7. Confidence Logic (Anomaly Detection based on structural deviation)
        dev = abs(pred_true - ml_components["mean_emission"])
        if dev > (3 * ml_components["std_emission"]):
            confidence = "Low (Anomaly Detected / Out of Distribution)"
        elif dev > (2 * ml_components["std_emission"]):
            confidence = "Medium"
        else:
            confidence = "High"

        # 8. Return strictly typed JSON object
        return EmissionResponse(
            predicted_emission=round(pred_true, 2),
            status_level=status,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        # Pydantic already rejected malformed input with a 422 before we got here,
        # so anything failing at this point is a server-side fault, not a bad request.
        print(f"[ERROR] Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
