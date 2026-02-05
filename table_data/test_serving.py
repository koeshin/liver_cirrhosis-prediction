import requests
import sys

url = "http://localhost:8000/predict"

# Sample Data (Stage 3 likely, or at least valid data)
payload = {
    "Age": 50,
    "Sex": "F",
    "Bilirubin": 1.0,
    "Albumin": 3.5,
    "Copper": 50,
    "Alk_Phos": 1000,
    "SGOT": 100,
    "Cholesterol": 300,
    "Tryglicerides": 100,
    "Platelets": 250,
    "Prothrombin": 10.0,
    "Ascites": "N",
    "Hepatomegaly": "N",
    "Spiders": "N",
    "Edema": "N"
}

print(f"Sending request to {url} with payload: {payload}")

try:
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print("\n✅ Success! Response received.")
        print("Status Code:", response.status_code)
        
        # Check if output contains expected HTML elements
        if "Result: Stage" in response.text:
             print("✅ Valid Prediction found in HTML.")
             # Extract Stage for verification
             start = response.text.find("Result: Stage")
             print("Extract:", response.text[start:start+20])
        else:
             print("⚠️ Warning: 'Result: Stage' not found in HTML output.")
             print(response.text[:500]) # Print first 500 chars
             
    else:
        print(f"❌ Failed. Status Code: {response.status_code}")
        print("Response:", response.text)

except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Is the server running on localhost:8000?")
except Exception as e:
    print(f"❌ An error occurred: {e}")
