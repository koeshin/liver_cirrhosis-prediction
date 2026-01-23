from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# 1. 저장된 모델 불러오기
# (주의: 이 모델은 아래의 파생변수들이 포함된 데이터로 '새로 학습된 모델'이어야 합니다!)
model = joblib.load("liver_model_FE_best.pkl")

app = FastAPI()

# 2. 입력 데이터 정의 (사용자는 '원본 수치'만 보냅니다)
class InputData(BaseModel):
    # 수치형 변수
    Bilirubin: float
    Cholesterol: float
    Albumin: float
    Copper: float
    Alk_Phos: float
    SGOT: float
    Tryglicerides: float
    Platelets: float
    Prothrombin: float
    
    # 범주형 및 기타 변수 (학습 때 사용한 것들)
    Sex: int            # 0 or 1
    Ascites: int        # 0 or 1
    Hepatomegaly: int   # 0 or 1
    Spiders: int        # 0 or 1
    Edema: int          # 0 or 1

@app.post("/predict")
def predict(data: InputData):
    # 1. 입력 데이터를 DataFrame으로 변환
    df = pd.DataFrame([data.dict()])
    
    # ---------------------------------------------------------
    # 2. [핵심] 서버 내부에서 피처 엔지니어링 수행 (Feat. Numpy)
    # ---------------------------------------------------------
    eps = 1e-6  # 0으로 나누기 방지용 아주 작은 수
    
    # (1) 단위 변환 (계산용 임시 변수)
    # bili mg/dL -> μmol/L, alb g/dL -> g/L
    df["bili_umolL"] = df["Bilirubin"] * 17.104
    df["alb_gL"] = df["Albumin"] * 10.0
    df["plt_1000uL"] = df["Platelets"] # 보통 입력 단위가 1000/μL라고 가정

    # (2) ALBI Score (알부민-빌리루빈 지수)
    # 공식: 0.66 * log10(bili) - 0.085 * alb
    df["ALBI"] = 0.66 * np.log10(df["bili_umolL"] + eps) - 0.085 * df["alb_gL"]

    # (3) PALBI Score (혈소판 포함 ALBI)
    log_bili = np.log10(df["bili_umolL"] + eps)
    log_plt  = np.log10(df["plt_1000uL"] + eps)
    df["PALBI"] = (2.02 * log_bili) - (0.37 * (log_bili**2)) \
                  - (0.04 * df["alb_gL"]) - (3.48 * log_plt) + (1.01 * (log_plt**2))

    # (4) APRI (AST to Platelet Ratio Index)
    # 공식: ((AST / ULN) * 100) / Platelets
    AST_ULN = 40.0  # 기준값
    df["APRI"] = ((df["SGOT"] / AST_ULN) * 100.0) / (df["plt_1000uL"] + eps)

    # (5) 파생 변수 (Interaction Features)
    # bili_x_albumin: 빌리루빈 / 알부민
    df["bili_x_albumin"] = df["Bilirubin"] / (df["Albumin"] + eps)

    # pt_x_bili: 프로트롬빈 * 빌리루빈 (응고지연 x 황달)
    df["pt_x_bili"] = df["Prothrombin"] * df["Bilirubin"]

    # portal_hint: (1 / Platelets) * Hepatomegaly (문맥압 항진 힌트)
    df["portal_hint"] = (1.0 / (df["Platelets"] + eps)) * df["Hepatomegaly"]

    # chol_over_alp: 콜레스테롤 / 알칼리성 인산분해효소
    df["chol_over_alp"] = df["Cholesterol"] / (df["Alk_Phos"] + eps)

    # (6) 계산에만 쓰고 모델에는 안 들어가는 임시 컬럼 삭제
    # (모델 학습 때 이 컬럼들이 없었다면 반드시 지워야 함)
    drop_cols = ["bili_umolL", "alb_gL", "plt_1000uL"]
    df = df.drop(columns=drop_cols, errors='ignore')

    # ---------------------------------------------------------
    # 3. 예측 수행
    # ---------------------------------------------------------
    try:
        # 모델에 들어가는 최종 컬럼 순서가 학습 때와 맞는지 확인하면 더 좋습니다.
        prediction_idx = model.predict(df)[0]
        predicted_stage = int(prediction_idx) + 1
        
        return {
            "predicted_stage": predicted_stage,
            "features_used": list(df.columns), # 디버깅용: 어떤 피처가 들어갔는지 확인
            "description": f"계산된 파생변수를 포함하여 예측한 결과: Stage {predicted_stage}"
        }
    except Exception as e:
        return {"error": str(e), "message": "모델의 입력 피처 개수와 생성된 피처 개수가 맞는지 확인하세요."}