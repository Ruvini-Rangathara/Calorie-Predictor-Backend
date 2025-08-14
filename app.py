from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from pathlib import Path
import traceback
from model_utils import get_predictor, reload_predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config["DEBUG"] = os.getenv("DEBUG", "False").lower() == "true"

@app.route("/")
def home():
    """API documentation endpoint"""
    return jsonify({
        "service": "Calorie Prediction API",
        "model": "CatBoost",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Single prediction", 
            "/batch_predict": "POST - Multiple predictions"
        },
        "required_fields": [
            "Sex", "Age", "Height", "Weight", 
            "Duration", "Heart_Rate", "Body_Temp"
        ],
        "example": {
            "Sex": "male",
            "Age": 35,
            "Height": 175.0,
            "Weight": 70.0,
            "Duration": 30.0,
            "Heart_Rate": 120.0,
            "Body_Temp": 39.5
        }
    })

@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        predictor = get_predictor()
        return jsonify({
            "status": "healthy",
            "model_loaded": predictor.model is not None,
            "preprocessor_loaded": predictor.preprocessor is not None,
            "model_path": str(predictor.model_path),
            "preprocessor_path": str(predictor.preprocessor_path)
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy", 
            "error": str(e)
        }), 500

@app.route("/predict", methods=["POST"])
def predict_calories():
    """Single calorie prediction endpoint"""
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided",
                "required_fields": [
                    "Sex", "Age", "Height", "Weight", 
                    "Duration", "Heart_Rate", "Body_Temp"
                ]
            }), 400

        logger.info(f"Prediction request: {data}")

        # Make prediction
        predictor = get_predictor()
        predicted_calories = predictor.predict(data)

        # Return response
        response = {
            "success": True,
            "input": data,
            "predicted_calories": round(predicted_calories, 2),
            "model": "CatBoost"
        }
        
        logger.info(f"Prediction successful: {predicted_calories:.2f} calories")
        return jsonify(response)

    except ValueError as e:
        # Input validation errors
        logger.warning(f"Validation error: {e}")
        return jsonify({
            "success": False,
            "error": "Invalid input data",
            "details": str(e)
        }), 400

    except Exception as e:
        # Unexpected errors
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e) if app.config["DEBUG"] else "Prediction failed"
        }), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Batch calorie prediction endpoint"""
    try:
        data = request.get_json()
        if not data or "inputs" not in data:
            return jsonify({
                "success": False,
                "error": "Expected JSON with 'inputs' array"
            }), 400

        inputs = data["inputs"]
        if not isinstance(inputs, list):
            return jsonify({
                "success": False,
                "error": "'inputs' must be an array"
            }), 400

        predictor = get_predictor()
        results = []

        # Process each input
        for i, input_data in enumerate(inputs):
            try:
                predicted_calories = predictor.predict(input_data)
                results.append({
                    "index": i,
                    "success": True,
                    "input": input_data,
                    "predicted_calories": round(predicted_calories, 2)
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "input": input_data,
                    "error": str(e)
                })

        # Calculate success rate
        successful = sum(1 for r in results if r["success"])
        
        return jsonify({
            "success": True,
            "total_inputs": len(inputs),
            "successful_predictions": successful,
            "failed_predictions": len(inputs) - successful,
            "results": results
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/reload", methods=["POST"])
def reload_model():
    """Reload model and preprocessor (useful for updates)"""
    try:
        predictor = reload_predictor()
        return jsonify({
            "success": True,
            "message": "Model and preprocessor reloaded successfully",
            "model_path": str(predictor.model_path),
            "preprocessor_path": str(predictor.preprocessor_path)
        })
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/predict", "/batch_predict", "/reload"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Verify required directories and files exist
    required_files = [
        Path("models/CatBoost_model.pkl"),
        Path("preprocessor/Preprocessor_CatBoost.pkl")
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.error("Missing required files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.error("Please ensure model and preprocessor files are in place")
        exit(1)

    # Initialize predictor on startup
    try:
        predictor = get_predictor()
        logger.info("✅ CatBoost model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model/preprocessor: {e}")
        exit(1)

    # Start server
    port = int(os.getenv("PORT", 5000))
    debug = app.config["DEBUG"]
    
    logger.info(f"🚀 Starting Calorie Prediction API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)