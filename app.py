from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import os
from pathlib import Path
import traceback
from model_utils import get_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['DEBUG'] = os.getenv('DEBUG', 'False').lower() == 'true'
app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', 'models/XGBoost_model.pkl')

@app.route('/')
def home():
    """Home page with API documentation"""
    return jsonify({
        "message": "Calorie Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": {
                "url": "/predict",
                "method": "POST",
                "description": "Predict calories burned",
                "required_fields": [
                    "Sex", "Age", "Height", "Weight", 
                    "Duration", "Heart_Rate", "Body_Temp"
                ]
            },
            "health": {
                "url": "/health",
                "method": "GET",
                "description": "Health check endpoint"
            }
        },
        "example_request": {
            "Sex": "male",
            "Age": 35,
            "Height": 175.0,
            "Weight": 70.0,
            "Duration": 30.0,
            "Heart_Rate": 120.0,
            "Body_Temp": 39.5
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        predictor = get_predictor()
        return jsonify({
            "status": "healthy",
            "model_loaded": predictor.model is not None,
            "model_path": str(predictor.model_path)
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict_calories():
    """Predict calories burned based on input parameters"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "required_fields": ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
            }), 400
        
        logger.info(f"Received prediction request: {data}")
        
        # Get predictor and make prediction
        predictor = get_predictor()
        predicted_calories = predictor.predict(data)
        
        # Prepare response
        response = {
            "success": True,
            "input": data,
            "predicted_calories": round(predicted_calories, 2),
            "model": "XGBoost",
            "message": f"Predicted {predicted_calories:.2f} calories burned"
        }
        
        logger.info(f"Prediction successful: {predicted_calories:.2f} calories")
        return jsonify(response)
        
    except ValueError as e:
        # Validation errors
        logger.warning(f"Validation error: {e}")
        return jsonify({
            "success": False,
            "error": "Validation error",
            "details": str(e)
        }), 400
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e) if app.config['DEBUG'] else "An error occurred during prediction"
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict calories for multiple inputs"""
    try:
        data = request.get_json()
        
        if not data or 'inputs' not in data:
            return jsonify({
                "error": "Expected JSON with 'inputs' array"
            }), 400
        
        inputs = data['inputs']
        if not isinstance(inputs, list):
            return jsonify({
                "error": "'inputs' must be an array"
            }), 400
        
        predictor = get_predictor()
        results = []
        
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
        
        return jsonify({
            "success": True,
            "total_inputs": len(inputs),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/predict", "/batch_predict"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Ensure model directory exists
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Check if model file exists
    model_path = model_dir / 'XGBoost_model.pkl'
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please place your XGBoost_model.pkl file in the models/ directory")
        exit(1)
    
    # Initialize predictor on startup
    try:
        predictor = get_predictor()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        exit(1)
    
    # Run the app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)