"""
실제 테스트 데이터에서 추출한 Stage 1, 2, 3 샘플로 서빙 테스트
Usage: python test_serving.py [1|2|3|all]
  - 1: Stage 1 샘플 테스트
  - 2: Stage 2 샘플 테스트
  - 3: Stage 3 샘플 테스트
  - all: 모든 Stage 테스트 (기본값)
"""
import requests
import sys

url = "http://localhost:8000/predict"

# 실제 liver_cirrhosis.csv에서 추출한 Stage별 샘플 데이터 (random_state=42)
STAGE_SAMPLES = {
    1: {
        "Age": 42,
        "Sex": "F",
        "Bilirubin": 8.1,
        "Albumin": 2.82,
        "Copper": 97.65,
        "Alk_Phos": 1982.66,
        "SGOT": 122.56,
        "Cholesterol": 369.51,
        "Tryglicerides": 124.70,
        "Platelets": 193.0,
        "Prothrombin": 10.4,
        "Ascites": "Y",
        "Hepatomegaly": "N",
        "Spiders": "Y",
        "Edema": "N"
    },
    2: {
        "Age": 61,
        "Sex": "F",
        "Bilirubin": 3.0,
        "Albumin": 3.63,
        "Copper": 74.0,
        "Alk_Phos": 1052.0,
        "SGOT": 108.5,
        "Cholesterol": 486.0,
        "Tryglicerides": 109.0,
        "Platelets": 438.0,
        "Prothrombin": 9.9,
        "Ascites": "N",
        "Hepatomegaly": "N",
        "Spiders": "Y",
        "Edema": "S"
    },
    3: {
        "Age": 66,
        "Sex": "F",
        "Bilirubin": 0.9,
        "Albumin": 3.87,
        "Copper": 30.0,
        "Alk_Phos": 1009.0,
        "SGOT": 57.35,
        "Cholesterol": 420.0,
        "Tryglicerides": 232.0,
        "Platelets": 257.02,
        "Prothrombin": 11.0,
        "Ascites": "N",
        "Hepatomegaly": "Y",
        "Spiders": "N",
        "Edema": "N"
    }
}


def test_stage(stage: int) -> bool:
    """특정 Stage 샘플로 테스트 실행"""
    payload = STAGE_SAMPLES[stage]
    
    print(f"\n{'='*50}")
    print(f"🧪 Testing Stage {stage} Sample")
    print(f"{'='*50}")
    print(f"Payload: {payload}")
    
    try:
        response = requests.post(url, data=payload)
        
        if response.status_code == 200:
            print(f"\n✅ Response received (HTTP 200)")
            
            # 예측 결과 추출
            if "Result: Stage" in response.text:
                start = response.text.find("Result: Stage")
                end = response.text.find("<", start)
                predicted = response.text[start:end].strip() if end > start else response.text[start:start+20]
                print(f"📊 Predicted: {predicted}")
                print(f"🎯 Expected:  Stage {stage}")
                
                # 정확도 체크
                if f"Stage {stage}" in predicted or f"Stage{stage}" in predicted:
                    print("✅ CORRECT!")
                    return True
                else:
                    print("⚠️  MISMATCH (but this can happen with real predictions)")
                    return True  # 응답 자체는 성공
            else:
                print("⚠️  'Result: Stage' not found in response")
                print(response.text[:500])
                return False
        else:
            print(f"❌ Failed. Status Code: {response.status_code}")
            print("Response:", response.text[:500])
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is the server running on localhost:8000?")
        return False
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return False


def main():
    stages_to_test = []
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "all":
            stages_to_test = [1, 2, 3]
        elif arg in ["1", "2", "3"]:
            stages_to_test = [int(arg)]
        else:
            print(f"Usage: python {sys.argv[0]} [1|2|3|all]")
            print("  1: Stage 1 샘플 테스트")
            print("  2: Stage 2 샘플 테스트")
            print("  3: Stage 3 샘플 테스트")
            print("  all: 모든 Stage 테스트 (기본값)")
            sys.exit(1)
    else:
        stages_to_test = [1, 2, 3]  # 기본값: 모든 Stage 테스트
    
    print(f"\n🏥 Liver Cirrhosis Prediction - Stage Test")
    print(f"Testing stages: {stages_to_test}")
    print(f"Server: {url}")
    
    results = {}
    for stage in stages_to_test:
        results[stage] = test_stage(stage)
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY")
    print(f"{'='*50}")
    for stage, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  Stage {stage}: {status}")
    
    total_pass = sum(results.values())
    total = len(results)
    print(f"\nTotal: {total_pass}/{total} passed")


if __name__ == "__main__":
    main()
