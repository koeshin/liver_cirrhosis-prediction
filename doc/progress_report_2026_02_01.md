# 간질환 진행 단계 예측 프로젝트 진행 보고서 (2026-02-01)

## 1. 개요
본 프로젝트는 간경변(Liver Cirrhosis) 환자의 데이터를 기반으로 질환의 상태 및 진행 단계를 예측하는 모델과 서빙 시스템을 개발하는 것을 목표로 합니다. 오늘까지 **Model B(진행 단계 예측)**의 고도화와 **서빙 시스템(FastAPI)**의 상용 수준 개선이 완료되었습니다.

## 2. 주요 성과

### ✅ Model B: Multi-class Progression Model 고도화
- **상태**: 3단계(Stage 1, 2, 3) 다중 분류 모델로 리팩토링 완료.
- **모델링**: Random Forest, XGBoost, LightGBM 및 정교한 Voting Ensemble 적용.
- **피처 엔지니어링**:
    - 의료적 지표(ALBI, APRI, FIB-4, PALBI 등) 자동 생성 로직 통합.
    - 변수별 로그 변환 및 최적화된 스케일링 적용.

### ✅ Model Serving System (FastAPI) 구축 및 개선
- **기능**: 웹 UI를 통해 실시간으로 환자 데이터를 입력하고 예측 결과를 확인 가능.
- **사용자 경험(UX) 개선**:
    - 나이 입력을 '일(Days)'에서 '세(Years)' 단위로 변경하여 편의성 증대.
    - 불필요한 입력 변수(`N_Days`) 제거.
    - 시각적 디자인을 Bootstrap 기반으로 깔끔하게 개선.
- **설명 가능한 AI (Explainable AI)**:
    - **동적 SHAP 분석**: 모든 추론마다 **가장 신뢰도(Confidence)가 높은 모델**을 자동으로 선택하여 SHAP Waterfall Plot 제공.
    - **지표 노출**: 하단 대시보드에 5개 주요 파생 의료 지표 실시간 계산 및 노출.

## 3. 작업 파일 현황
- `table_data/model_b_progression.ipynb`: 고도화된 Model B 학습 노트북.
- `table_data/serving.py`: FastAPI 기반 웹 서버 스크립트.
- `table_data/saved_models/`: 최적화된 학습 모델 가중치 파일들.
- `table_data/cirrhosis.csv`: 원본 데이터셋.

## 4. 향후 계획
- **데이터 증강**: 불균형한 클래스(Stage 1, 2) 데이터 보정 및 추가 확보.
- **Model A 통합**: 간경변 진단(Model A)과 진행 단계 예측(Model B)을 하나의 서빙 시스템으로 통합.
- **모델 성능 고도화**: 더 깊은 하이퍼파라미터 튜닝 및 딥러닝 기반 모델과의 비교 연구.

---
**기록일**: 2026년 02월 01일
**작성자**: Antigravity (AI Pair Programmer)
