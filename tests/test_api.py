import os
import sys
import requests
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_api():
    """Test the FastAPI endpoints"""
    BASE_URL = "http://localhost:8000"
    
    print("🧪 Testing FastAPI Endpoints...")
    
    try:
        # 1. Test root endpoint
        print("1. Testing root endpoint...")
        response = requests.get(f"{BASE_URL}/")
        print(f"   ✅ Root: {response.json()}")
        
        # 2. Test health check
        print("2. Testing health check...")
        response = requests.get(f"{BASE_URL}/health")
        print(f"   ✅ Health: {response.json()}")
        
        # 3. Test examples endpoint
        print("3. Testing examples endpoint...")
        response = requests.get(f"{BASE_URL}/examples")
        print(f"   ✅ Examples: {len(response.json()['example_queries'])} examples")
        
        # 4. Test stats endpoint
        print("4. Testing stats endpoint...")
        response = requests.get(f"{BASE_URL}/stats")
        stats = response.json()
        print(f"   ✅ Stats: {stats['system']['name']} - {stats['system']['version']}")
        
        # 5. Test query endpoint
        print("5. Testing query endpoint...")
        query_data = {
            "question": "How do I create a FastAPI endpoint?",
            "filters": None
        }
        response = requests.post(f"{BASE_URL}/query", json=query_data)
        result = response.json()
        print(f"   ✅ Query: '{result['question']}'")
        print(f"   📝 Answer: {result['answer'][:100]}...")
        print(f"   🎯 Confidence: {result['confidence']}")
        print(f"   ⚡ Processing time: {result['processing_time']}s")
        
        print("\n🎉 ALL API TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("💡 Make sure the server is running: python src/api.py")
        return False

if __name__ == "__main__":
    test_api()