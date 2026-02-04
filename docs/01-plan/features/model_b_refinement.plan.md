# Model B Refinement - Plan Document

## 개요
- **목표**: Model B(Stage 1-3)의 신뢰성, 설명 가능성, 재현성을 강화하여 의료 현장 및 기술 면접에서의 방어 논리를 완성함.
- **배경**: 현재 구조에서 모델 선택의 정당성 부족, 설명의 불안정성, 범위 밖 데이터 처리에 대한 위험이 존재함.

## 범위
### 포함
- **의사결정 범위/정책**: Stage 1-3 및 Uncertain(불확실) 정의
- **동적 선택 정당화**: Calibration + Confidence 기반 모델 선택
- **설명 일관성**: 임상 개념 그룹(Clinical Concept Groups) 단위의 SHAP 제공
- **재현성**: 결정 로그(Decision Logs) 및 메타데이터 기록

### 제외
- 새로운 모델 아키텍처 탐색 (기존 모델 재활용)
- 프론트엔드 UI 수정 (API/노트북 레벨까지가 범위, 단 JSON 스키마는 정의)

## 성공 기준
- [ ] 기준 1: OOD(Out of Distribution) 또는 낮은 Confidence 입력시 'Uncertain' 출력 확인
- [ ] 기준 2: Calibration 적용 후 ECE 감소 확인 (Comparison Report)
- [ ] 기준 3: 동일 입력에 대해 항상 동일한 Feature Group Importance 출력 (Consistency)
- [ ] 기준 4: 동일 입력에 대해 항상 동일한 결정 로그 생성 (Reproducibility)

## 일정
| 단계 | 예상 기간 | 비고 |
|-----|---------|-----|
| 설계 | 10분 | PDCA Design Agent |
| 구현 | 20분 | PDCA Implementer Agent |
| 검증 | 10분 | PDCA Gap Analyzer Agent |

## 리스크
| 리스크 | 영향도 | 발생확률 | 대응 방안 |
|-------|-------|--------|---------|
| Calibration 데이터 부족 | 높음 | 중간 | Hold-out set의 크기를 확인하고 Cross-validation 활용 고려 |
| SHAP 계산 속도 저하 | 중간 | 높음 | Background dataset 축소 또는 TreeExplainer 최적화 |

## 의존성
- 기존 `model_b_progression.ipynb` 파일
- `shap`, `scikit-learn` 라이브러리

## 이해관계자
- 담당자: Antigravity
- 검토자: User
