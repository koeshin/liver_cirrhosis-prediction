# Decision Policy for Liver Cirrhosis Prediction (Model B)

## 1. Output Classes
The model predicts the progression stage of liver cirrhosis.
- **Valid Stages**: Stage 1, Stage 2, Stage 3.
- **Out of Scope**: Stage 4 (Terminal/Transplant) is explicitly excluded from this model's training and inference.

## 2. Uncertainty Handling
To ensure reliability in clinical settings, the model implements an "Uncertainty" mechanism.

- **Criteria**:
  If the maximum predicted probability (confidence) < **0.6 (60%)**, the prediction is flagged as **Uncertain**.
  
- **Action**:
  - **Sufficient Confidence**: Return `Predicted Stage`.
  - **Uncertain**: Return `Uncertain` status with a reason (e.g., "Confidence 0.45 < 0.6").
  
## 3. Calibration
- All probabilities are post-processed using **Isotonic Regression** (or Platt Scaling) to ensure that predicted probabilities match observed frequencies.
- Raw model outputs are NOT used for final decision making.

## 4. Explanation Policy (SHAP)
- Explanations are aggregated into **Clinical Concept Groups** to reduce noise and improve consistency.
    - **Liver Biomarkers**: Bilirubin, Albumin, Alk_Phos, SGOT, Prothrombin, Copper
    - **Vitals**: Platelets, Cholesterol
    - **Demographics**: Age, Sex
    - **Clinical Signs**: Ascites, Hepatomegaly, Spiders, Edema
    - **Treatment**: Drug
