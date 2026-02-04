
# 2026-02-04 작업 기록

## 요약
- 총 작업 수: 1개 (Model B Critical Refinement)
- 총 변경 파일: 2개 (Notebook, Utils) + Docs
- 총 변경 라인: +200 / -50 (Estimated)

---

## 작업 1: Model B Workflow Refinement (Round 1-3)

### 무엇을
- 대상: `model_b_pdca.ipynb`, `model_b_pdca_utils.py`
- 범위: Model B의 동적 선택(Dynamic Selection) 및 캘리브레이션 워크플로우 전반

### 어떻게
- 방법: PDCA 반복(Plan-Do-Check-Act)을 통해 5가지 핵심 개선 사항 적용
- 도구: `scikit-learn` (Sigmoid Calibration, Train/Test Split), `shap`, `numpy` (Clipping)

### 왜
- 배경: 초기 구현에서 확률 붕괴(Probability Collapse), 데이터 누수(Leakage), 불확실성 정책 미작동 문제 발견
- 목적: 면접 방어 가능한 수준의 견고한(Robust) 워크플로우 구축

### 결과
1. **Data Leakage 해결**: `Stage`, `N_Days` 피처 제거 (입력에서 Target 배제)
2. **3-Way Split 구현**: Train(60%) / Calib(20%) / Test(20%) 분리 (Test 오염 방지)
3. **Confidence Stabilization**: `Sigmoid` 캘리브레이션 전환 + `eps` 클리핑으로 확률 붕괴(0.9998) 해결
4. **Dynamic Selection 고도화**: 단순 Max Conf 대신 **Margin Score** 도입 + 'Uncertain' 정책 정상화
5. **SHAP Integration**: 자동 설명 생성 파이프라인 구축 (빈 설명 방지)
6. **Validation**: p90/p95/p99 지표 추가로 분포 건전성 확인 가능

---
