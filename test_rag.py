import requests
import json

# Base URL for your Flask app (remove trailing slash)
base_url = "https://portfoliobackend-ubbd.onrender.com"

def test_endpoint():
    try:
        # Test the health check endpoint
        response = requests.get(f"{base_url}/")
        print("\nHealth check response:", response.text)
        print("Status code:", response.status_code)
        
        # Test questions about Kishore
        test_questions = [
            "What are Kishore's internships?",
            "Tell me about Kishore's achievements",
            "What technologies does Kishore work with?"
        ]

        # Test each question
        for question in test_questions:
            print(f"\nAsking: {question}")
            try:
                response = requests.post(
                    f"{base_url}/query", 
                    json={"question": question},
                    headers={"Content-Type": "application/json"}
                )
                print("Status code:", response.status_code)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'error' in result:
                        print("Error:", result['error'])
                    else:
                        print("Answer:", result['answer'])
                        print("Context used:", result['context'])
                else:
                    print("Response text:", response.text)
            except requests.exceptions.RequestException as e:
                print(f"Error making request: {e}")
            except json.JSONDecodeError as e:
                print(f"Error decoding response: {e}")
                print("Response text:", response.text)
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_endpoint() 