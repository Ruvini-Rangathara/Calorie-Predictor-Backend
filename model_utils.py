import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CaloriePredictor:
    """Fixed Calorie Predictor using CatBoost model and preprocessor"""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.model = None
        self.preprocessor = None
        self.required_features = [
            'Sex', 'Age', 'Height', 'Weight', 
            'Duration', 'Heart_Rate', 'Body_Temp'
        ]
        # Training statistics for proper feature engineering
        self.training_stats = None
        self._load_components()
    
    def _load_components(self):
        """Load model and preprocessor from pickle files"""
        try:
            # Load model
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Load preprocessor - Note: this might be redundant since the model is a pipeline
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
            
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info(f"Preprocessor loaded successfully from {self.preprocessor_path}")
            
            # Set training statistics for feature engineering (from the notebook analysis)
            self.training_stats = {
                'bmi_mean': 24.5,  # Approximate from typical BMI distribution
                'bmi_std': 4.0     # Approximate standard deviation
            }
            
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            raise
    
    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean input data"""
        # Check required fields
        missing_fields = [field for field in self.required_features if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Create validated data dictionary
        validated_data = {}
        
        # Validate Sex - match the exact format from training
        sex = str(data['Sex']).lower().strip()
        if sex in ['male', 'm']:
            validated_data['Sex'] = 'Male'  # Exact case from training data
        elif sex in ['female', 'f']:
            validated_data['Sex'] = 'Female'  # Exact case from training data
        else:
            raise ValueError(f"Invalid Sex value: {data['Sex']}. Must be 'male' or 'female'")
        
        # Validate numeric fields with ranges
        numeric_validations = {
            'Age': (1, 120),
            'Height': (50, 250),  # cm
            'Weight': (20, 300),  # kg
            'Duration': (1, 1440), # minutes (max 24 hours)
            'Heart_Rate': (30, 220), # bpm
            'Body_Temp': (30, 45)  # Celsius
        }
        
        for field, (min_val, max_val) in numeric_validations.items():
            try:
                value = float(data[field])
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{field} must be between {min_val} and {max_val}, got {value}")
                validated_data[field] = value
            except (ValueError, TypeError):
                raise ValueError(f"Invalid {field} value: {data[field]}. Must be a number.")
        
        return validated_data
    
    def _engineer_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature engineering exactly as done in training"""
        df = pd.DataFrame([data])
        
        # Create BMI feature (as done in the notebook)
        df['BMI'] = df['Weight'] / (df['Height']/100)**2
        
        # Ensure the column order matches what the model expects
        # From the notebook: the model expects these exact columns in this order
        expected_columns = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
        df = df[expected_columns]
        
        return df
    
    def predict(self, input_data: Dict[str, Any]) -> float:
        """Make calorie prediction with proper pipeline handling"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Validate input
            validated_data = self._validate_input(input_data)
            logger.info(f"Input validated: {validated_data}")
            
            # Apply feature engineering
            df = self._engineer_features(validated_data)
            logger.info(f"Features engineered, shape: {df.shape}, columns: {list(df.columns)}")
            
            # The model is a pipeline that includes preprocessing, so we just pass the raw DataFrame
            # The model pipeline will handle all preprocessing internally
            log_prediction = self.model.predict(df)[0]
            logger.info(f"Log prediction: {log_prediction}")
            
            # Convert from log space to original space
            prediction = np.expm1(log_prediction)
            
            # Ensure positive prediction
            prediction = max(prediction, 0.01)
            
            logger.info(f"Final prediction: {prediction:.2f} calories")
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

# Global predictor instance
_predictor: Optional[CaloriePredictor] = None

def get_predictor() -> CaloriePredictor:
    """Get singleton predictor instance"""
    global _predictor
    if _predictor is None:
        model_path = "models/CatBoost_model.pkl"
        preprocessor_path = "preprocessor/Preprocessor_CatBoost.pkl"
        _predictor = CaloriePredictor(model_path, preprocessor_path)
    return _predictor

def reload_predictor():
    """Reload the predictor (useful for updates)"""
    global _predictor
    _predictor = None
    return get_predictor()