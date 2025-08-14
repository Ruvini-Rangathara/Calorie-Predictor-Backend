import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"

def test_home():
    """Test home endpoint with API documentation"""
    print("🏠 Testing home endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Home endpoint working")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
    print("-" * 60)

def test_health():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Health Status: {data.get('status')}")
        print(f"Model Loaded: {data.get('model_loaded')}")
        print(f"Preprocessor Loaded: {data.get('preprocessor_loaded')}")
        
        if data.get('status') == 'healthy':
            print("✅ Service is healthy")
        else:
            print(f"⚠️ Service unhealthy: {data.get('error')}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
    print("-" * 60)

def test_single_prediction():
    """Test single calorie prediction"""
    print("🔥 Testing single prediction...")
    
    test_cases = [
        {
            "name": "Male, moderate workout",
            "data": {
                "Sex": "male",
                "Age": 35,
                "Height": 175.0,
                "Weight": 70.0,
                "Duration": 30.0,
                "Heart_Rate": 120.0,
                "Body_Temp": 39.5
            }
        },
        {
            "name": "Female, intense workout", 
            "data": {
                "Sex": "female",
                "Age": 28,
                "Height": 165.0,
                "Weight": 55.0,
                "Duration": 45.0,
                "Heart_Rate": 160.0,
                "Body_Temp": 40.2
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 Test: {test_case['name']}")
        try:
            response = requests.post(f"{BASE_URL}/predict", json=test_case['data'])
            data = response.json()
            
            print(f"Status: {response.status_code}")
            if data.get('success'):
                print(f"✅ Predicted Calories: {data['predicted_calories']}")
                print(f"Input: {data['input']}")
            else:
                print(f"❌ Prediction failed: {data.get('error')}")
                if 'details' in data:
                    print(f"Details: {data['details']}")
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
    
    print("-" * 60)

def test_batch_prediction():
    """Test batch prediction"""
    print("📦 Testing batch prediction...")
    
    batch_data = {
        "inputs": [
            {
                "Sex": "male",
                "Age": 25,
                "Height": 180.0,
                "Weight": 75.0,
                "Duration": 60.0,
                "Heart_Rate": 140.0,
                "Body_Temp": 39.8
            },
            {
                "Sex": "female",
                "Age": 32,
                "Height": 160.0,
                "Weight": 58.0,
                "Duration": 25.0,
                "Heart_Rate": 110.0,
                "Body_Temp": 39.2
            },
            {
                "Sex": "male",
                "Age": 45,
                "Height": 185.0,
                "Weight": 85.0,
                "Duration": 40.0,
                "Heart_Rate": 130.0,
                "Body_Temp": 39.7
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/batch_predict", json=batch_data)
        data = response.json()
        
        print(f"Status: {response.status_code}")
        if data.get('success'):
            print(f"✅ Batch prediction successful")
            print(f"Total inputs: {data['total_inputs']}")
            print(f"Successful: {data['successful_predictions']}")
            print(f"Failed: {data['failed_predictions']}")
            
            print("\nResults:")
            for result in data['results']:
                if result['success']:
                    print(f"  #{result['index']}: {result['predicted_calories']} calories")
                else:
                    print(f"  #{result['index']}: Failed - {result['error']}")
        else:
            print(f"❌ Batch prediction failed: {data.get('error')}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    
    print("-" * 60)

def test_error_handling():
    """Test various error conditions"""
    print("⚠️ Testing error handling...")
    
    error_tests = [
        {
            "name": "Missing fields",
            "data": {"Sex": "male", "Age": 30}
        },
        {
            "name": "Invalid sex value",
            "data": {
                "Sex": "unknown",
                "Age": 35,
                "Height": 175.0,
                "Weight": 70.0,
                "Duration": 30.0,
                "Heart_Rate": 120.0,
                "Body_Temp": 39.5
            }
        },
        {
            "name": "Invalid age (negative)",
            "data": {
                "Sex": "male",
                "Age": -5,
                "Height": 175.0,
                "Weight": 70.0,
                "Duration": 30.0,
                "Heart_Rate": 120.0,
                "Body_Temp": 39.5
            }
        },
        {
            "name": "Invalid heart rate (too high)",
            "data": {
                "Sex": "female",
                "Age": 25,
                "Height": 165.0,
                "Weight": 60.0,
                "Duration": 30.0,
                "Heart_Rate": 250.0,  # Too high
                "Body_Temp": 39.0
            }
        }
    ]
    
    for test in error_tests:
        print(f"\n🧪 Testing: {test['name']}")
        try:
            response = requests.post(f"{BASE_URL}/predict", json=test['data'])
            data = response.json()
            
            print(f"Status: {response.status_code}")
            if response.status_code == 400:
                print(f"✅ Properly rejected: {data.get('error')}")
            else:
                print(f"⚠️ Unexpected response: {data}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
    
    print("-" * 60)

def test_performance():
    """Test API performance"""
    print("⚡ Testing API performance...")
    
    test_data = {
        "Sex": "male",
        "Age": 30,
        "Height": 175.0,
        "Weight": 70.0,
        "Duration": 30.0,
        "Heart_Rate": 120.0,
        "Body_Temp": 39.5
    }
    
    num_requests = 5
    times = []
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            response = requests.post(f"{BASE_URL}/predict", json=test_data)
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
                print(f"Request {i+1}: {(end_time - start_time)*1000:.2f}ms")
            else:
                print(f"Request {i+1}: Failed ({response.status_code})")
                
        except requests.exceptions.RequestException as e:
            print(f"Request {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📈 Performance Summary:")
        print(f"Average response time: {avg_time*1000:.2f}ms")
        print(f"Min: {min(times)*1000:.2f}ms")
        print(f"Max: {max(times)*1000:.2f}ms")
    
    print("-" * 60)

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting Comprehensive API Tests")
    print("=" * 60)
    
    try:
        # Basic functionality tests
        test_home()
        test_health()
        test_single_prediction()
        test_batch_prediction()
        
        # Error handling tests
        test_error_handling()
        
        # Performance test
        test_performance()
        
        print("✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error!")
        print("Make sure the Flask server is running:")
        print("  python app.py")
        print("Then run the tests again.")
    except Exception as e:
        print(f"❌ Test suite error: {e}")

if __name__ == "__main__":
    run_all_tests()