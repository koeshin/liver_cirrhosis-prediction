# Model B Refinement - Design Document

## 개요
Plan 문서 참조: [model_b_refinement.plan.md](../../01-plan/features/model_b_refinement.plan.md)

## 아키텍처

### 시스템 구조
Notebook 기반의 파이프라인으로, 다음과 같은 순차적 처리를 수행함.
Input -> [Preprocessing] -> [Model Candidates (Stage 1, 2, 3)] -> [Calibration] -> [Dynamic Selection] -> [Uncertainty Check] -> [SHAP Explainer (Grouped)] -> Output (Prediction + Explanation + Logs)

### 기술 스택
| 영역 | 기술 | 선택 이유 |
|-----|-----|---------|
| Core | Python, Jupyter | 기존 환경 유지 |
| ML | Scikit-learn, XGBoost/LightGBM | 기존 모델 호환 및 Calibration 지원 |
| XAI | SHAP | 설명력 제공 |

## 데이터 모델

### Output Schema (JSON)
```json
{
  "prediction": {
    "stage": "integer (1, 2, 3) or null",
    "label": "string (Stage 1, Stage 2, Stage 3, Uncertain)",
    "confidence": "float (0.0 - 1.0)"
  },
  "explanation": {
    "top_groups": [
      {
        "name": "string (e.g., Liver Function)",
        "score": "float (impact)",
        "items": ["list of feature names"]
      }
    ]
  },
  "meta_logs": {
    "timestamp": "ISO8601",
    "selected_model": "string",
    "model_versions": {
      "stage_1": "v1.0",
      "stage_2": "v1.0",
      "stage_3": "v1.0"
    },
    "calibration_score": "float",
    "uncertain_reason": "string (optional)"
  }
}
```

## 알고리즘 상세 설계

### 1. Decision Policy & Uncertainty
- **Input**: `p_cal` (Calibrated Probability vector)
- **Logic**:
  - `max_prob = max(p_cal)`
  - `IF max_prob < THRESHOLD (e.g., 0.6) THEN Status = Uncertain`
  - `ELSE Status = Predicted Class`

### 2. Calibration & Dynamic Selection
- **Calibration**: Use `CalibratedClassifierCV` (method='isotonic' or 'sigmoid') on validation set.
- **Selection**: 
  - 각 Stage별 Binary Classifier들이 있다고 가정 (또는 Multi-class).
  - 본 과제에서는 "가장 확신하는 모델"을 선택하는 것이 아니라, "단일 Multi-class 모델"에서 "가장 높은 확률의 클래스"를 신뢰하거나, 
  - 혹은 "Stage별 1vsRest 모델들" 중에서 `max(calibrated_prob)`가 가장 높은 모델을 승자로 선택.
  - **Design Decision**: `model_b`는 Progression 예측이므로 Multi-label/Multi-class 성격임. 기존 로직을 분석하여 "Stage별 예측 확률"을 보정하는 방식 채택.

### 3. Grouped SHAP
- **Concept Groups**:
  - `Liver Biomarkers`: Bilirubin, Albumin, Alkaline_Phosphatase, SGOT, Prothrombin
  - `Vitals`: Platelets, Cholesterol, Tryglicerides
  - `Demographics/History`: Age, Sex, Ascites, Hepatomegaly, Spiders, Edema
  - `Drugs`: Drug (D-penicillamine vs Placebo)
- **Calculation**:
  - `Group_SHAP = sum(|SHAP_feature_i|) for feature_i in Group` (절대값 합 또는 단순 합, 여기서는 기여도의 크기가 중요하므로 절대값 합 권장하지만, 방향성 유지를 위해 단순 합 사용 후 시각화 시 처리) -> **Policy**: 단순 합(Sum)으로 방향성 유지.

## 구현 순서

1. [ ] 1단계: `model_b_pdca.ipynb` 생성 및 데이터 로드 verification
2. [ ] 2단계: Calibration 로직 구현 및 전후 비교 (ECE Score)
3. [ ] 3단계: Uncertainty Thresholding 및 Decision Logic 구현
4. [ ] 4단계: Grouped SHAP 구현
5. [ ] 5단계: Logging 구조체 및 최종 Pipeline 함수 생성

## 에러 핸들링
| 상황 | 처리 방식 |
|-----|---------|
| SHAP 계산 실패 | Explanation 필드에 "Not Available" 명시 후 예측값만 리턴 |
| Input feature 누락 | Preprocessing 단계에서 Default 값 채움 or Error Log |
