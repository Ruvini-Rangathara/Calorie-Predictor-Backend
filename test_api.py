import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    print("-" * 50)

def test_single_prediction():
    """Test single prediction"""
    print("🔍 Testing single prediction...")
    
    test_data = {
        "Sex": "male",
        "Age": 35,
        "Height": 175.0,
        "Weight": 70.0,
        "Duration": 30.0,
        "Heart_Rate": 120.0,
        "Body_Temp": 39.5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    print("-" * 50)

def test_batch_prediction():
    """Test batch prediction"""
    print("🔍 Testing batch prediction...")
    
    test_data = {
        "inputs": [
            {
                "Sex": "male",
                "Age": 25,
                "Height": 180.0,
                "Weight": 75.0,
                "Duration": 45.0,
                "Heart_Rate": 140.0,
                "Body_Temp": 40.0
            },
            {
                "Sex": "female",
                "Age": 30,
                "Height": 165.0,
                "Weight": 60.0,
                "Duration": 20.0,
                "Heart_Rate": 110.0,
                "Body_Temp": 39.0
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/batch_predict", json=test_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    print("-" * 50)

def test_error_handling():
    """Test error handling"""
    print("🔍 Testing error handling...")
    
    # Test missing fields
    incomplete_data = {
        "Sex": "male",
        "Age": 35
        # Missing other required fields
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=incomplete_data)
        print(f"Missing fields - Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    print("-" * 25)
    
    # Test invalid values
    invalid_data = {
        "Sex": "invalid",
        "Age": -5,
        "Height": 175.0,
        "Weight": 70.0,
        "Duration": 30.0,
        "Heart_Rate": 120.0,
        "Body_Temp": 39.5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        print(f"Invalid values - Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    print("-" * 50)

def test_home_endpoint():
    """Test home endpoint"""
    print("🔍 Testing home endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    print("-" * 50)

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting API Tests...")
    print("=" * 50)
    
    try:
        test_home_endpoint()
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        print("✅ All tests completed!")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the server is running on http://localhost:5000")
        print("Run 'python app.py' in another terminal first.")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    run_all_tests()