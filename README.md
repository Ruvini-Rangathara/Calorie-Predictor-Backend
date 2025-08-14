# Calorie Prediction API - Kaggle Playground Series S5E5

## Overview
This project is the backend component for the [Kaggle Playground Series S5E5](https://www.kaggle.com/competitions/playground-series-s5e5/overview) competition. The Calorie Prediction API is a Flask-based RESTful service that provides calorie burn predictions using a CatBoost machine learning model. It supports single and batch predictions based on input features such as gender, age, height, weight, exercise duration, heart rate, and body temperature. The API is designed to integrate with a React-based frontend and includes comprehensive error handling, logging, and testing scripts.

## Features
- **RESTful Endpoints**: Supports health checks, single predictions, batch predictions, and model reloading.
- **CatBoost Model**: Utilizes a pre-trained CatBoost model for accurate calorie burn predictions.
- **Input Validation**: Ensures all input data meets strict validation criteria.
- **Feature Engineering**: Includes BMI calculation and data preprocessing consistent with training.
- **Error Handling**: Robust error responses for invalid inputs and server issues.
- **Logging**: Detailed logging for debugging and monitoring.
- **Testing Suite**: Comprehensive test script for verifying API functionality and performance.

## Tech Stack
- **Backend**: Flask, Flask-CORS
- **Machine Learning**: CatBoost, scikit-learn, pandas, numpy, joblib
- **Deployment**: Gunicorn (optional for production)
- **Testing**: Requests library for API testing
- **Dependencies**: Managed via `requirements.txt`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd calorie-prediction-api
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the root directory with the following:
   ```env
   DEBUG=False
   PORT=5000
   MODEL_PATH=models/CatBoost_model.pkl
   PREPROCESSOR_PATH=preprocessor/Preprocessor_CatBoost.pkl
   LOG_LEVEL=INFO
   ```

5. **Ensure Model Files**:
   - Place the pre-trained CatBoost model at `models/CatBoost_model.pkl`.
   - Place the preprocessor at `preprocessor/Preprocessor_CatBoost.pkl`.
   - These files are required for the API to function.

6. **Run the API**:
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:5000`.

## API Endpoints
| Endpoint | Method | Description | Request Body (JSON) Example |
|----------|--------|-------------|----------------------------|
| `/` | GET | API documentation | N/A |
| `/health` | GET | Health check for model and preprocessor | N/A |
| `/predict` | POST | Single calorie prediction | `{"Sex": "male", "Age": 35, "Height": 175.0, "Weight": 70.0, "Duration": 30.0, "Heart_Rate": 120.0, "Body_Temp": 39.5}` |
| `/batch_predict` | POST | Batch calorie predictions | `{"inputs": [{"Sex": "male", "Age": 25, ...}, ...]}` |
| `/reload` | POST | Reload model and preprocessor | N/A |

### Required Input Fields
- `Sex`: String ("male" or "female")
- `Age`: Number (1–120 years)
- `Height`: Number (50–250 cm)
- `Weight`: Number (20–300 kg)
- `Duration`: Number (1–1440 minutes)
- `Heart_Rate`: Number (30–220 bpm)
- `Body_Temp`: Number (30–45 °C)

## Project Structure
```
calorie-prediction-api/
├── models/
│   └── CatBoost_model.pkl         # Pre-trained CatBoost model
├── preprocessor/
│   └── Preprocessor_CatBoost.pkl  # Preprocessor for feature engineering
├── app.py                        # Main Flask application
├── model_utils.py                # CaloriePredictor class and utilities
├── test_api.py                   # Test script for API endpoints
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
└── README.md                     # Project documentation
```

## Usage
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Test the API using the provided test script:
   ```bash
   python test_api.py
   ```
3. Make requests using a tool like `curl`, Postman, or the frontend application:
   ```bash
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"Sex":"male","Age":35,"Height":175.0,"Weight":70.0,"Duration":30.0,"Heart_Rate":120.0,"Body_Temp":39.5}'
   ```
4. Check logs for debugging and monitoring:
   - Logs are output to the console and can be configured via `LOG_LEVEL` in `.env`.

## Testing
The `test_api.py` script includes tests for:
- Home endpoint (`/`)
- Health check (`/health`)
- Single prediction (`/predict`)
- Batch prediction (`/batch_predict`)
- Error handling for invalid inputs
- Performance testing (response times)

Run the tests:
```bash
python test_api.py
```

## Notes
- **Model and Preprocessor**: The API expects `CatBoost_model.pkl` and `Preprocessor_CatBoost.pkl` to be present in the `models/` and `preprocessor/` directories, respectively.
- **Feature Engineering**: The API calculates BMI and ensures the feature order matches the training pipeline.
- **Error Handling**: Invalid inputs return a 400 status code with detailed error messages. Server errors return a 500 status code.
- **Deployment**: For production, consider using Gunicorn:
  ```bash
  gunicorn --bind 0.0.0.0:5000 app:app
  ```
- **Dataset**: The API is designed for the Kaggle Playground Series S5E5 dataset, which includes the required features listed above.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Built for the [Kaggle Playground Series S5E5](https://www.kaggle.com/competitions/playground-series-s5e5/overview) competition.
- Powered by CatBoost for accurate calorie predictions.
- Designed to integrate with a React frontend for a complete user experience.