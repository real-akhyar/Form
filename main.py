# inference_api.py
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional
import warnings

# Remove warnings
warnings.filterwarnings("ignore")

# Load trained models and encoder
try:
    rf_model = joblib.load("Random_Forest_Model.pkl")
    tree_model = joblib.load("Tree_Model.pkl")
    knn_model = joblib.load("KNN_Model.pkl")
    encoder = joblib.load("encoder.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}. Please ensure all .pkl files are in the same directory.")

# Initialize FastAPI app
app = FastAPI(
    title="Human Activity Recognition API",
    description="API for predicting human activities from accelerometer data",
    version="1.0.0"
)

# Pydantic model for input validation
class SensorData(BaseModel):
    timestamp: int
    x_axis: float
    y_axis: float
    z_axis: float
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": 4991922345000,
                "x_axis": 0.69,
                "y_axis": 10.8,
                "z_axis": -2.03
            }
        }

class BatchSensorData(BaseModel):
    data: List[SensorData]
    
    @validator('data')
    def validate_data_length(cls, v):
        if len(v) == 0:
            raise ValueError("Data list cannot be empty")
        return v

class PredictionRequest(BaseModel):
    data: BatchSensorData
    model_type: Optional[str] = "random_forest"  # Options: random_forest, decision_tree, knn, all
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ["random_forest", "decision_tree", "knn", "all"]
        if v not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
        return v

class PredictionResult(BaseModel):
    activity: str
    activity_code: int
    confidence: Optional[float] = None
    model_used: str

class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    model_type: str
    total_samples: int

class ModelComparisonResult(BaseModel):
    random_forest: BatchPredictionResult
    decision_tree: BatchPredictionResult
    knn: BatchPredictionResult

def create_features_for_knn(data_df: pd.DataFrame) -> pd.DataFrame:
    """Create KNN features as done during training"""
    df = data_df.copy()
    
    # Create squared acceleration
    df["sq_acc"] = df["x_axis"]**2 + df["y_axis"]**2 + df["z_axis"]**2
    
    # Rolling statistics
    for axis in ['x_axis', 'y_axis', 'z_axis']:
        df[f'{axis}_mean'] = df[axis].rolling(window=3, min_periods=1).mean()
        df[f'{axis}_std'] = df[axis].rolling(window=3, min_periods=1).std()
    
    # Squared acceleration statistics
    df['sq_mean'] = df['sq_acc'].rolling(window=3, min_periods=1).mean()
    df['sq_std'] = df['sq_acc'].rolling(window=3, min_periods=1).std()
    
    # Exponential moving averages
    for axis in ['x_axis', 'y_axis', 'z_axis']:
        df[f'{axis}_ema'] = df[axis].ewm(span=3, min_periods=1).mean()
    
    # Fill NaN values with column means
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Select features in the same order as training
    knn_features = [
        "timestamp", "x_axis", "y_axis", "z_axis", "sq_acc",
        "x_axis_std", "x_axis_mean", "y_axis_std", "y_axis_mean",
        "z_axis_std", "z_axis_mean", "sq_mean", "sq_std",
        "x_axis_ema", "y_axis_ema", "z_axis_ema"
    ]
    
    return df[knn_features]

def predict_random_forest(features_df: pd.DataFrame) -> BatchPredictionResult:
    """Make predictions using Random Forest model"""
    # Prepare features (same as training)
    X = features_df[['timestamp', 'x_axis', 'y_axis', 'z_axis']].values
    
    # Make predictions
    predictions = rf_model.predict(X)
    
    # Get probabilities if available
    if hasattr(rf_model, 'predict_proba'):
        probabilities = rf_model.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
    else:
        confidences = [None] * len(predictions)
    
    # Decode activity labels
    activities = encoder.inverse_transform(predictions)
    
    # Prepare results
    results = []
    for i, (pred, act) in enumerate(zip(predictions, activities)):
        results.append(PredictionResult(
            activity=act,
            activity_code=int(pred),
            confidence=float(confidences[i]) if confidences[i] is not None else None,
            model_used="random_forest"
        ))
    
    return BatchPredictionResult(
        predictions=results,
        model_type="random_forest",
        total_samples=len(results)
    )

def predict_decision_tree(features_df: pd.DataFrame) -> BatchPredictionResult:
    """Make predictions using Decision Tree model"""
    # Prepare features (same as training)
    X = features_df[['timestamp', 'x_axis', 'y_axis', 'z_axis']].values
    
    # Make predictions
    predictions = tree_model.predict(X)
    
    # Get probabilities if available
    if hasattr(tree_model, 'predict_proba'):
        probabilities = tree_model.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
    else:
        confidences = [None] * len(predictions)
    
    # Decode activity labels
    activities = encoder.inverse_transform(predictions)
    
    # Prepare results
    results = []
    for i, (pred, act) in enumerate(zip(predictions, activities)):
        results.append(PredictionResult(
            activity=act,
            activity_code=int(pred),
            confidence=float(confidences[i]) if confidences[i] is not None else None,
            model_used="decision_tree"
        ))
    
    return BatchPredictionResult(
        predictions=results,
        model_type="decision_tree",
        total_samples=len(results)
    )

def predict_knn(features_df: pd.DataFrame) -> BatchPredictionResult:
    """Make predictions using KNN model"""
    # Create KNN features
    knn_features_df = create_features_for_knn(features_df)
    
    # Make predictions
    predictions = knn_model.predict(knn_features_df.values)
    
    # Get probabilities if available
    if hasattr(knn_model, 'predict_proba'):
        probabilities = knn_model.predict_proba(knn_features_df.values)
        confidences = np.max(probabilities, axis=1)
    else:
        confidences = [None] * len(predictions)
    
    # Decode activity labels
    activities = encoder.inverse_transform(predictions)
    
    # Prepare results
    results = []
    for i, (pred, act) in enumerate(zip(predictions, activities)):
        results.append(PredictionResult(
            activity=act,
            activity_code=int(pred),
            confidence=float(confidences[i]) if confidences[i] is not None else None,
            model_used="knn"
        ))
    
    return BatchPredictionResult(
        predictions=results,
        model_type="knn",
        total_samples=len(results)
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Human Activity Recognition API",
        "version": "1.0.0",
        "available_models": ["random_forest", "decision_tree", "knn"],
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /models": "List available models",
            "POST /predict": "Make predictions"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": True}

@app.get("/models")
async def list_models():
    """List available models and their information"""
    models_info = {
        "random_forest": {
            "type": "RandomForestClassifier",
            "n_estimators": getattr(rf_model, 'n_estimators', 'unknown'),
            "classes": encoder.classes_.tolist()
        },
        "decision_tree": {
            "type": "DecisionTreeClassifier",
            "max_depth": getattr(tree_model, 'max_depth', 'unknown'),
            "classes": encoder.classes_.tolist()
        },
        "knn": {
            "type": "KNeighborsClassifier",
            "n_neighbors": getattr(knn_model, 'n_neighbors', 'unknown'),
            "classes": encoder.classes_.tolist()
        }
    }
    return models_info

@app.post("/predict", response_model=ModelComparisonResult)
async def predict(request: PredictionRequest):
    """
    Predict human activities from sensor data
    
    - **model_type**: Which model to use (random_forest, decision_tree, knn, all)
    - **data**: List of sensor readings with timestamp, x_axis, y_axis, z_axis
    """
    try:
        # Convert input data to DataFrame
        data_list = [{"timestamp": d.timestamp, 
                     "x_axis": d.x_axis, 
                     "y_axis": d.y_axis, 
                     "z_axis": d.z_axis} 
                    for d in request.data.data]
        
        features_df = pd.DataFrame(data_list)
        
        # Make predictions based on model_type
        if request.model_type == "random_forest":
            rf_result = predict_random_forest(features_df)
            return ModelComparisonResult(
                random_forest=rf_result,
                decision_tree=BatchPredictionResult(predictions=[], model_type="decision_tree", total_samples=0),
                knn=BatchPredictionResult(predictions=[], model_type="knn", total_samples=0)
            )
        
        elif request.model_type == "decision_tree":
            tree_result = predict_decision_tree(features_df)
            return ModelComparisonResult(
                random_forest=BatchPredictionResult(predictions=[], model_type="random_forest", total_samples=0),
                decision_tree=tree_result,
                knn=BatchPredictionResult(predictions=[], model_type="knn", total_samples=0)
            )
        
        elif request.model_type == "knn":
            knn_result = predict_knn(features_df)
            return ModelComparisonResult(
                random_forest=BatchPredictionResult(predictions=[], model_type="random_forest", total_samples=0),
                decision_tree=BatchPredictionResult(predictions=[], model_type="decision_tree", total_samples=0),
                knn=knn_result
            )
        
        else:  # "all" - compare all models
            rf_result = predict_random_forest(features_df)
            tree_result = predict_decision_tree(features_df)
            knn_result = predict_knn(features_df)
            
            return ModelComparisonResult(
                random_forest=rf_result,
                decision_tree=tree_result,
                knn=knn_result
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/single", response_model=PredictionResult)
async def predict_single(data: SensorData, model_type: str = "random_forest"):
    """
    Predict activity for a single sensor reading
    """
    try:
        # Create batch with single item
        batch_request = PredictionRequest(
            data=BatchSensorData(data=[data]),
            model_type=model_type
        )
        
        # Get prediction
        result = await predict(batch_request)
        
        # Extract single result based on model type
        if model_type == "random_forest":
            return result.random_forest.predictions[0]
        elif model_type == "decision_tree":
            return result.decision_tree.predictions[0]
        elif model_type == "knn":
            return result.knn.predictions[0]
        else:
            # For "all", return random forest result by default
            return result.random_forest.predictions[0]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)