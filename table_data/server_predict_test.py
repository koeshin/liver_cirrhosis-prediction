import requests
import json

url = "http://127.0.0.1:8000/predict"

patient_data = {
    "Bilirubin": 14.5, "Cholesterol": 261.0, "Albumin": 2.6, "Copper": 156.0,
    "Alk_Phos": 1718.0, "SGOT": 137.95, "Tryglicerides": 172.0, "Platelets": 190.0,
    "Prothrombin": 12.2, "Sex": 0, "Ascites": 1, "Hepatomegaly": 1, "Spiders": 1, "Edema": 1
}

try:
    response = requests.post(url, json=patient_data)
    
    # 1. 일단 서버가 준 전체 내용을 찍어봅니다 (디버깅용)
    print("📩 서버 응답 원본:", response.json()) 

    if response.status_code == 200:
        result = response.json()
        
        # 2. 에러가 있는지 먼저 검사
        if "error" in result:
            print("\n❌ 서버 내부 에러:", result["error"])
            print("👉 메시지:", result.get("message", ""))
        else:
            print("\n✅ 예측 성공!")
            print(f"▶ 예측 단계: Stage {result['predicted_stage']}")

    else:
        print("\n❌ 통신 에러:", response.text)

except Exception as e:
    print("\n⚠️ 클라이언트 실행 중 에러:", e)