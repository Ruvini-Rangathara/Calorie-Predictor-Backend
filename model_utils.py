import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
import pickle
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaloriePredictor:
    def __init__(self, model_path="models/XGBoost_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.train_stats = None
        self.feature_columns = None
        self._initialize_stats()
        self._load_model()

    def _fix_xgboost_compatibility(self, model):
        """Enhanced XGBoost compatibility fix"""
        try:
            import xgboost as xgb

            if hasattr(model, "steps"):  # Pipeline
                for step_name, step_model in model.steps:
                    if (
                        hasattr(step_model, "__class__")
                        and "XGB" in step_model.__class__.__name__
                    ):
                        self._fix_xgb_model_attributes(step_model)
                        # Additional fix for sklearn tags
                        self._patch_sklearn_tags(step_model)
            elif hasattr(model, "__class__") and "XGB" in model.__class__.__name__:
                self._fix_xgb_model_attributes(model)
                self._patch_sklearn_tags(model)

            return model
        except Exception as e:
            logger.warning(f"Could not apply XGBoost compatibility fix: {e}")
            return model

    def _patch_sklearn_tags(self, xgb_model):
        """Patch sklearn tags compatibility issue"""
        try:
            # Override the problematic __sklearn_tags__ method
            def safe_sklearn_tags(self):
                return {
                    "requires_fit": True,
                    "requires_positive_X": False,
                    "requires_y": True,
                    "X_types": ["2darray"],
                    "y_types": ["1dlabels"],
                    "allow_nan": False,
                    "stateless": False,
                    "binary_only": False,
                    "_xfail_checks": {},
                }

            # Monkey patch the method
            xgb_model.__sklearn_tags__ = safe_sklearn_tags.__get__(
                xgb_model, xgb_model.__class__
            )
            logger.info("XGBoost sklearn tags patched successfully")

        except Exception as e:
            logger.warning(f"Could not patch sklearn tags: {e}")

    def _fix_xgb_model_attributes(self, xgb_model):
        """Fix missing XGBoost model attributes"""
        missing_attrs = {
            "gpu_id": None,
            "device": "cpu",
            "enable_categorical": False,
            "feature_types": None,
            "max_cat_to_onehot": None,
            "max_cat_threshold": None,
            "eval_metric": None,
            "early_stopping_rounds": None,
            "callbacks": None,
            "interaction_constraints": None,
            "monotone_constraints": None,
            "predictor": "auto",
            "n_jobs": 1,
            "verbosity": 0,
        }

        for attr, default_value in missing_attrs.items():
            if not hasattr(xgb_model, attr):
                setattr(xgb_model, attr, default_value)

        logger.info("XGBoost model attributes fixed for compatibility")

    def _load_model_with_fallback(self):
        """Load model with multiple fallback methods"""
        try:
            # Method 1: Try joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = joblib.load(self.model_path)
            logger.info("Model loaded successfully with joblib")
            return model
        except Exception as e1:
            logger.warning(f"Joblib loading failed: {e1}")

            try:
                # Method 2: Try pickle
                with open(self.model_path, "rb") as f:
                    model = pickle.load(f)
                logger.info("Model loaded successfully with pickle")
                return model
            except Exception as e2:
                logger.warning(f"Pickle loading failed: {e2}")

                try:
                    # Method 3: Try XGBoost native loading if it's a standalone XGBoost model
                    import xgboost as xgb

                    if str(self.model_path).endswith(".json") or str(
                        self.model_path
                    ).endswith(".ubj"):
                        model = xgb.XGBRegressor()
                        model.load_model(self.model_path)
                        logger.info(
                            "Model loaded successfully with XGBoost native loader"
                        )
                        return model
                except Exception as e3:
                    logger.error(f"All loading methods failed. Last error: {e3}")
                    raise e1  # Raise the original joblib error

    def _load_model(self):
        """Load and prepare the model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Load model with fallback methods
            self.model = self._load_model_with_fallback()

            # Apply compatibility fixes
            self.model = self._fix_xgboost_compatibility(self.model)

            # Try to extract feature names
            if hasattr(self.model, "named_steps"):
                preprocessor = self.model.named_steps.get("preprocessor")
                if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
                    try:
                        self.feature_columns = preprocessor.get_feature_names_out()
                        logger.info(
                            f"Feature columns extracted: {len(self.feature_columns)} features"
                        )
                    except Exception as e:
                        logger.warning(f"Could not extract feature names: {e}")

            # Test the model with a dummy prediction to catch issues early
            self._test_model()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _test_model(self):
        """Test model with dummy data to catch compatibility issues early"""
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
            columns_to_remove = ["id", "Calories", "Calories_log", "Duration_Group"]
            df_features = df_engineered.drop(
                columns=[
                    col for col in columns_to_remove if col in df_engineered.columns
                ]
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Test prediction
                prediction = self.model.predict(df_features)
                logger.info("Model test prediction successful")

        except Exception as e:
            logger.error(f"Model test failed: {e}")
            # Don't raise here, let the actual prediction handle it
            pass

    def _initialize_stats(self):
        """Initialize training statistics for missing value imputation"""
        self.train_stats = {
            "Age": {"median": 35.0},
            "Height": {"median": 175.0},
            "Weight": {"median": 70.0},
            "Duration": {"median": 20.0},
            "Heart_Rate": {"median": 95.0},
            "Body_Temp": {"median": 40.0},
            "Sex": {"mode": "male"},
        }
        logger.info("Training statistics initialized")

    def clean_data(self, df):
        """Clean and preprocess input data"""
        df = df.copy()

        # Handle numeric columns
        numeric_cols = [
            "Age",
            "Height",
            "Weight",
            "Duration",
            "Heart_Rate",
            "Body_Temp",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(self.train_stats[col]["median"])

        # Handle categorical columns
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].fillna(self.train_stats["Sex"]["mode"])
            df["Sex"] = df["Sex"].str.lower().replace({"m": "male", "f": "female"})

        return df

    def engineer_features(self, df):
        """Create engineered features"""
        df = df.copy()

        # BMI calculation
        if "Weight" in df.columns and "Height" in df.columns:
            df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2

        # Activity intensity
        if "Duration" in df.columns and "Heart_Rate" in df.columns:
            df["Activity_Intensity"] = df["Duration"] * df["Heart_Rate"]

        # Metabolic age
        if "Age" in df.columns and "BMI" in df.columns:
            df["Metabolic_Age"] = df["Age"] * df["BMI"] / 10

        # Cardio effort
        if (
            "Heart_Rate" in df.columns
            and "Duration" in df.columns
            and "Age" in df.columns
        ):
            df["Cardio_Effort"] = df["Heart_Rate"] * df["Duration"] / (df["Age"] + 1)

        # Temperature to heart rate ratio
        if "Body_Temp" in df.columns and "Heart_Rate" in df.columns:
            df["Temp_HR_Ratio"] = df["Body_Temp"] / df["Heart_Rate"]

        # Weight to height ratio
        if "Weight" in df.columns and "Height" in df.columns:
            df["Weight_Height_Ratio"] = df["Weight"] / (df["Height"] + 1)

        # Age groups
        if "Age" in df.columns:
            df["Age_Group"] = pd.cut(
                df["Age"],
                bins=[0, 30, 50, 100],
                labels=["<30", "30-50", ">50"],
                right=False,
            )

        # Duration groups
        if "Duration" in df.columns:
            df["Duration_Group"] = pd.cut(
                df["Duration"],
                bins=[0, 10, 20, 31],
                labels=["Short", "Medium", "Long"],
                right=True,
            )

        return df

    def validate_input(self, data):
        """Validate input data"""
        required_fields = [
            "Sex",
            "Age",
            "Height",
            "Weight",
            "Duration",
            "Heart_Rate",
            "Body_Temp",
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Value range validations
        validations = {
            "Age": (1, 120, "Age must be between 1 and 120"),
            "Height": (50, 250, "Height must be between 50 and 250 cm"),
            "Weight": (20, 300, "Weight must be between 20 and 300 kg"),
            "Duration": (1, 300, "Duration must be between 1 and 300 minutes"),
            "Heart_Rate": (40, 220, "Heart rate must be between 40 and 220 bpm"),
            "Body_Temp": (35, 45, "Body temperature must be between 35 and 45°C"),
        }

        errors = []
        for field, (min_val, max_val, message) in validations.items():
            try:
                value = float(data[field])
                if not (min_val <= value <= max_val):
                    errors.append(message)
            except (ValueError, TypeError):
                errors.append(f"{field} must be a valid number")

        # Sex validation
        if data.get("Sex", "").lower() not in ["male", "female", "m", "f"]:
            errors.append("Sex must be 'male' or 'female'")

        if errors:
            raise ValueError("; ".join(errors))

        return True

    def predict(self, input_data):
        """Make calorie prediction"""
        try:
            # Validate input
            self.validate_input(input_data)

            # Convert to DataFrame if needed
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()

            # Process data
            df_clean = self.clean_data(df)
            df_engineered = self.engineer_features(df_clean)

            # Remove columns that shouldn't be in features
            columns_to_remove = ["id", "Calories", "Calories_log", "Duration_Group"]
            df_features = df_engineered.drop(
                columns=[
                    col for col in columns_to_remove if col in df_engineered.columns
                ]
            )

            # Make prediction with enhanced error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    log_prediction = self.model.predict(df_features)
                except Exception as e:
                    logger.error(f"Model prediction failed: {e}")
                    # Try alternative prediction methods
                    if hasattr(self.model, "named_steps"):
                        # If it's a pipeline, try to predict step by step
                        transformed_data = df_features
                        for step_name, step_model in self.model.named_steps.items():
                            if (
                                step_name != list(self.model.named_steps.keys())[-1]
                            ):  # Not the last step
                                transformed_data = step_model.transform(
                                    transformed_data
                                )
                            else:  # Last step (predictor)
                                log_prediction = step_model.predict(transformed_data)
                    else:
                        raise e

            # Convert from log space to actual calories
            calories_prediction = np.expm1(log_prediction)

            # Return single value or list
            if len(calories_prediction) == 1:
                return float(calories_prediction[0])
            return calories_prediction.tolist()

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# Singleton pattern for Flask
predictor = None


def get_predictor():
    """Get or create predictor instance"""
    global predictor
    if predictor is None:
        predictor = CaloriePredictor()
    return predictor
