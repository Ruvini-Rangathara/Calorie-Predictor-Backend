import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
import pickle
from catboost import CatBoostRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaloriePredictor:
    def __init__(self, model_path="models/CatBoost_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.train_stats = None
        self._initialize_stats()
        self._load_model()

    def _load_model(self):
        """Load and prepare the model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Try loading with joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = joblib.load(self.model_path)
            logger.info("Model loaded with joblib")
        except Exception as e1:
            logger.warning(f"Joblib load failed: {e1}")
            try:
                # Fallback to pickle
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded with pickle")
            except Exception as e2:
                logger.error(f"Failed to load model with both joblib and pickle: {e2}")
                raise e2

        # Test the model with a dummy prediction
        self._test_model()

    def _test_model(self):
        try:
            dummy_data = {
                "Sex": "male",
                "Age": 35,
                "Height": 175.0,
                "Weight": 70.0,
                "Duration": 30.0,
                "Heart_Rate": 120.0,
                "Body_Temp": 39.5,
            }

            df = pd.DataFrame([dummy_data])
            df_clean = self.clean_data(df)
            df_engineered = self.engineer_features(df_clean)
            df_features = df_engineered.drop(
                columns=[col for col in ["id", "Calories", "Calories_log", "Duration_Group"] if col in df_engineered.columns]
            )

            self.model.predict(df_features)
            logger.info("Model test prediction successful")
        except Exception as e:
            logger.error(f"Model test failed: {e}")

    def _initialize_stats(self):
        self.train_stats = {
            "Age": {"median": 35.0},
            "Height": {"median": 175.0},
            "Weight": {"median": 70.0},
            "Duration": {"median": 20.0},
            "Heart_Rate": {"median": 95.0},
            "Body_Temp": {"median": 40.0},
            "Sex": {"mode": "male"},
        }

    def clean_data(self, df):
        df = df.copy()
        numeric_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(self.train_stats[col]["median"])

        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].fillna(self.train_stats["Sex"]["mode"])
            df["Sex"] = df["Sex"].str.lower().replace({"m": "male", "f": "female"})

        return df

    def engineer_features(self, df):
        df = df.copy()

        if "Weight" in df.columns and "Height" in df.columns:
            df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2

        if "Duration" in df.columns and "Heart_Rate" in df.columns:
            df["Activity_Intensity"] = df["Duration"] * df["Heart_Rate"]

        if "Age" in df.columns and "BMI" in df.columns:
            df["Metabolic_Age"] = df["Age"] * df["BMI"] / 10

        if "Heart_Rate" in df.columns and "Duration" in df.columns and "Age" in df.columns:
            df["Cardio_Effort"] = df["Heart_Rate"] * df["Duration"] / (df["Age"] + 1)

        if "Body_Temp" in df.columns and "Heart_Rate" in df.columns:
            df["Temp_HR_Ratio"] = df["Body_Temp"] / df["Heart_Rate"]

        if "Weight" in df.columns and "Height" in df.columns:
            df["Weight_Height_Ratio"] = df["Weight"] / (df["Height"] + 1)

        if "Age" in df.columns:
            df["Age_Group"] = pd.cut(
                df["Age"], bins=[0, 30, 50, 100], labels=["<30", "30-50", ">50"], right=False
            )

        if "Duration" in df.columns:
            df["Duration_Group"] = pd.cut(
                df["Duration"], bins=[0, 10, 20, 31], labels=["Short", "Medium", "Long"], right=True
            )

        return df

    def validate_input(self, data):
        required_fields = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        validations = {
            "Age": (1, 120, "Age must be between 1 and 120"),
            "Height": (50, 250, "Height must be between 50 and 250 cm"),
            "Weight": (20, 300, "Weight must be between 20 and 300 kg"),
            "Duration": (1, 300, "Duration must be between 1 and 300 minutes"),
            "Heart_Rate": (40, 220, "Heart rate must be between 40 and 220 bpm"),
            "Body_Temp": (35, 45, "Body temperature must be between 35 and 45°C"),
        }

        errors = []
        for field, (min_val, max_val, msg) in validations.items():
            try:
                val = float(data[field])
                if not (min_val <= val <= max_val):
                    errors.append(msg)
            except:
                errors.append(f"{field} must be a number")

        if data.get("Sex", "").lower() not in ["male", "female", "m", "f"]:
            errors.append("Sex must be 'male' or 'female'")

        if errors:
            raise ValueError("; ".join(errors))
        return True

    def predict(self, input_data):
        try:
            self.validate_input(input_data)

            df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
            df_clean = self.clean_data(df)
            df_engineered = self.engineer_features(df_clean)
            df_features = df_engineered.drop(
                columns=[col for col in ["id", "Calories", "Calories_log", "Duration_Group"] if col in df_engineered.columns]
            )

            log_prediction = self.model.predict(df_features)
            calories_prediction = np.expm1(log_prediction)

            if len(calories_prediction) == 1:
                return float(calories_prediction[0])
            return calories_prediction.tolist()

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# Singleton for Flask or similar
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = CaloriePredictor()
    return predictor
