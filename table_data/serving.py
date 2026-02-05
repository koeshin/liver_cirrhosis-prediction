
import pandas as pd
import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shap
import matplotlib.pyplot as plt
import io
import base64
import os
import json
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI
app = FastAPI(title="Liver Cirrhosis Progression Prediction (Hierarchical)")

# Global Variables
sys_s3 = None
sys_s2 = None

def load_system(base_path):
    out_dir = f"{base_path}/artifacts/final_models"
    
    if not os.path.exists(f"{out_dir}/model_meta.json"):
        print(f"Warning: Model artifacts not found in {out_dir}")
        return None
        
    with open(f"{out_dir}/model_meta.json", 'r') as f:
        meta = json.load(f)
        
    scaler = joblib.load(f"{out_dir}/scaler.pkl")
    
    models = {}
    for m_name in ['rf', 'xgb', 'lgbm']:
        p = f"{out_dir}/{m_name}.pkl"
        if os.path.exists(p):
            models[m_name] = joblib.load(p)
            
    return {
        'meta': meta,
        'scaler': scaler,
        'models': models
    }

def load_models():
    global sys_s3, sys_s2
    print("Loading Hierarchical Systems...")
    
    sys_s3 = load_system("S12_vs_S3")
    sys_s2 = load_system("S1_vs_S2")
    
    if sys_s3 and sys_s2:
        print("✅ Systems loaded successfully")
        print(f"  S3 Threshold: {sys_s3['meta']['threshold']:.4f}")
        print(f"  S2 Threshold: {sys_s2['meta']['threshold']:.4f}")
    else:
        print("❌ Failed to load one or both systems.")

def create_score_feature(X, meta, score_meta):
    weights = np.array(score_meta['weights'])
    top3_cols = score_meta['top3_cols']
    
    if len(weights) == 0:
        return np.zeros(len(X))
        
    score_mean = np.array(score_meta['score_scaler_mean'])
    score_scale = np.array(score_meta['score_scaler_scale'])
    
    # Handle single sample DataFrame
    # Ensure columns exist
    missing = [c for c in top3_cols if c not in X.columns]
    if missing:
        print(f"Warning: Missing columns for score: {missing}")
        return np.zeros(len(X))
        
    X_sub = X[top3_cols].values
    X_sub_s = (X_sub - score_mean) / score_scale
    scores = np.dot(X_sub_s, weights)
    return scores

def prepare_inference_data(system, X_df):
    meta = system['meta']
    scaler = system['scaler']
    
    # 1. Feature Prep
    # Create Score Feature
    score_vals = create_score_feature(X_df, meta, meta['score_meta'])
    X_df_scored = X_df.copy()
    X_df_scored['top3_score'] = score_vals
    
    # One Hot (Naive matching)
    X_ohe = pd.get_dummies(X_df_scored, drop_first=True)
    
    # Align Columns
    final_cols = meta['features']
    for c in final_cols:
        if c not in X_ohe.columns:
            X_ohe[c] = 0
            
    X_ready = X_ohe[final_cols]
    
    # 2. Scale
    X_scaled = pd.DataFrame(scaler.transform(X_ready), columns=final_cols)
    
    return X_scaled, X_ready # Return scaled for model, ready (unscaled OHE) for nothing? actually SHAP needs transformed usually

def get_avg_prob(system, X_scaled):
    probs = np.zeros(len(X_scaled))
    models = system['models']
    count = 0
    for m in models.values():
        probs += m.predict_proba(X_scaled)[:, 1]
        count += 1
    return probs / count if count > 0 else np.zeros(len(X_scaled))

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Hierarchical Liver Cirrhosis Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f8f9fa; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .stage-box { padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;}
        .stage-1 { background-color: #28a745; }
        .stage-2 { background-color: #ffc107; color: black; }
        .stage-3 { background-color: #dc3545; }
        .prob-bar { height: 25px; border-radius: 5px; margin-top: 5px;}
        .test-btn-group { margin-bottom: 15px; }
        .test-btn-group .btn { margin-right: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">🩺 Liver Cirrhosis Hierarchical Prediction</h1>
        
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-dark text-white">Patient Data</div>
                    <div class="card-body">
                        <!-- Test Data Buttons -->
                        <div class="test-btn-group">
                            <label class="form-label fw-bold">📋 Load Test Data:</label>
                            <div>
                                <button type="button" class="btn btn-success btn-sm" onclick="loadStage1()">Stage 1</button>
                                <button type="button" class="btn btn-warning btn-sm" onclick="loadStage2()">Stage 2</button>
                                <button type="button" class="btn btn-danger btn-sm" onclick="loadStage3()">Stage 3</button>
                            </div>
                        </div>
                        <hr>
                        
                        <form id="predictionForm" action="/predict" method="post">
                            <!-- Inputs same as before, condensed -->
                            <div class="mb-2"><label>Age</label><input type="number" id="Age" name="Age" class="form-control" value="49"></div>
                            <div class="mb-2"><label>Bilirubin</label><input type="number" step="0.1" id="Bilirubin" name="Bilirubin" class="form-control" value="21.6"></div>
                            <div class="mb-2"><label>Albumin</label><input type="number" step="0.1" id="Albumin" name="Albumin" class="form-control" value="3.31"></div>
                            <div class="mb-2"><label>Copper</label><input type="number" step="0.01" id="Copper" name="Copper" class="form-control" value="221.0"></div>
                            <div class="mb-2"><label>Alk_Phos</label><input type="number" step="0.01" id="Alk_Phos" name="Alk_Phos" class="form-control" value="3697.4"></div>
                            <div class="mb-2"><label>SGOT</label><input type="number" step="0.01" id="SGOT" name="SGOT" class="form-control" value="101.91"></div>
                            <div class="mb-2"><label>Cholesterol</label><input type="number" step="0.01" id="Cholesterol" name="Cholesterol" class="form-control" value="175.0"></div>
                            <div class="mb-2"><label>Tryglicerides</label><input type="number" step="0.01" id="Tryglicerides" name="Tryglicerides" class="form-control" value="168.0"></div>
                            <div class="mb-2"><label>Platelets</label><input type="number" step="0.01" id="Platelets" name="Platelets" class="form-control" value="80.0"></div>
                            <div class="mb-2"><label>Prothrombin</label><input type="number" step="0.1" id="Prothrombin" name="Prothrombin" class="form-control" value="12.0"></div>
                            
                            <div class="mb-2"><label>Sex</label>
                                <select id="Sex" name="Sex" class="form-select">
                                    <option value="F" selected>Female</option>
                                    <option value="M">Male</option>
                                </select>
                            </div>
                            <div class="mb-2"><label>Ascites</label>
                                <select id="Ascites" name="Ascites" class="form-select">
                                    <option value="N">No</option>
                                    <option value="Y" selected>Yes</option>
                                </select>
                            </div>
                            <div class="mb-2"><label>Hepatomegaly</label>
                                <select id="Hepatomegaly" name="Hepatomegaly" class="form-select">
                                    <option value="N">No</option>
                                    <option value="Y" selected>Yes</option>
                                </select>
                            </div>
                            <div class="mb-2"><label>Spiders</label>
                                <select id="Spiders" name="Spiders" class="form-select">
                                    <option value="N">No</option>
                                    <option value="Y" selected>Yes</option>
                                </select>
                            </div>
                            <div class="mb-2"><label>Edema</label>
                                <select id="Edema" name="Edema" class="form-select">
                                    <option value="N">No</option>
                                    <option value="S" selected>Edema (No Diuretics)</option>
                                    <option value="Y">Edema (With Diuretics)</option>
                                </select>
                            </div>

                            <button type="submit" class="btn btn-primary w-100 mt-3">Predict</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <script>
            // liver_cirrhosis_deduped.csv에서 추출한 Stage별 샘플 데이터 (random_state=42)
            const stageData = {
                1: { Age: 52, Sex: "F", Bilirubin: 0.5, Albumin: 4.04, Copper: 227.0, Alk_Phos: 598.0, SGOT: 52.7, Cholesterol: 149.0, Tryglicerides: 57.0, Platelets: 421.0, Prothrombin: 9.9, Ascites: "N", Hepatomegaly: "Y", Spiders: "Y", Edema: "N" },
                2: { Age: 46, Sex: "F", Bilirubin: 0.4, Albumin: 3.03, Copper: 97.65, Alk_Phos: 1982.66, SGOT: 122.56, Cholesterol: 369.51, Tryglicerides: 124.7, Platelets: 173.0, Prothrombin: 10.9, Ascites: "Y", Hepatomegaly: "N", Spiders: "Y", Edema: "N" },
                3: { Age: 49, Sex: "F", Bilirubin: 21.6, Albumin: 3.31, Copper: 221.0, Alk_Phos: 3697.4, SGOT: 101.91, Cholesterol: 175.0, Tryglicerides: 168.0, Platelets: 80.0, Prothrombin: 12.0, Ascites: "Y", Hepatomegaly: "Y", Spiders: "Y", Edema: "S" }
            };
            
            function loadStageData(stage) {
                const data = stageData[stage];
                for (const [key, value] of Object.entries(data)) {
                    const el = document.getElementById(key);
                    if (el) el.value = value;
                }
                // Visual feedback
                alert('Stage ' + stage + ' 테스트 데이터가 로드되었습니다!');
            }
            
            function loadStage1() { loadStageData(1); }
            function loadStage2() { loadStageData(2); }
            function loadStage3() { loadStageData(3); }
            </script>
            
            <div class="col-md-8">
                {% if result %}
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4>Result: Stage {{ result.final_stage }}</h4>
                    </div>
                    <div class="card-body">
                         <div class="stage-box stage-{{ result.final_stage }}">
                            <h1>Stage {{ result.final_stage }}</h1>
                            <p class="mb-0">Determined via Hierarchical Cascade</p>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Step 1: S12 vs S3</h5>
                                <p>Probability of Stage 3: <strong>{{ "%.1f"|format(result.prob_s3 * 100) }}%</strong></p>
                                <p>(Threshold: {{ "%.2f"|format(result.th_s3) }})</p>
                                <div class="progress prob-bar">
                                    <div class="progress-bar bg-danger" style="width:{{ result.prob_s3 * 100 }}%">S3</div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <h5>Step 2: S1 vs S2</h5>
                                <p>Probability of Stage 2 (if not S3): <strong>{{ "%.1f"|format(result.prob_s2 * 100) }}%</strong></p>
                                <p>(Threshold: {{ "%.2f"|format(result.th_s2) }})</p>
                                <div class="progress prob-bar">
                                    <div class="progress-bar bg-warning" style="width:{{ result.prob_s2 * 100 }}%">S2</div>
                                </div>
                            </div>
                        </div>

                        <hr>
                        <h5>Estimated Probabilities</h5>
                        <ul>
                             <li>P(Stage 3) = {{ "%.1f"|format(result.final_probs.p3 * 100) }}%</li>
                             <li>P(Stage 2) = {{ "%.1f"|format(result.final_probs.p2 * 100) }}%</li>
                             <li>P(Stage 1) = {{ "%.1f"|format(result.final_probs.p1 * 100) }}%</li>
                        </ul>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h4>🔍 SHAP Analysis (LGBM)</h4>
                    </div>
                    <div class="card-body text-center">
                        <p>Explaining validation for: <strong>{{ result.shap_target }}</strong></p>
                        {% if shap_image %}
                        <img src="data:image/png;base64,{{ shap_image }}" class="img-fluid" alt="SHAP">
                        {% endif %}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
"""

# Routes
@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/", response_class=HTMLResponse)
def read_root():
    from jinja2 import Template
    t = Template(html_template)
    return t.render(result=None)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  Age: float = Form(...), Bilirubin: float = Form(...), 
                  Albumin: float = Form(...), Copper: float = Form(...), 
                  Alk_Phos: float = Form(...), SGOT: float = Form(...), 
                  Cholesterol: float = Form(...), Tryglicerides: float = Form(...), 
                  Platelets: float = Form(...), Prothrombin: float = Form(...), 
                  Edema: str = Form(...),
                  Sex: str = Form("F"), Ascites: str = Form("N"), 
                  Hepatomegaly: str = Form("N"), Spiders: str = Form("N")):
    
    # Feature Engineering (Same as before)
    input_data = {
        'Age': Age, 'Bilirubin': Bilirubin, 'Albumin': Albumin,
        'Copper': Copper, 'Alk_Phos': Alk_Phos, 'SGOT': SGOT, 
        'Cholesterol': Cholesterol, 'Tryglicerides': Tryglicerides, 
        'Platelets': Platelets, 'Prothrombin': Prothrombin,
        # Default categorical logic if inputs missing in form, but we pass them
        'Sex': Sex, 'Ascites': Ascites, 'Hepatomegaly': Hepatomegaly, 
        'Spiders': Spiders, 'Edema': Edema
    }
    
    df = pd.DataFrame([input_data])
    # ... Apply FE (simplified copy from before) ...
    df['Age_Year'] = df['Age'] / 365.25 # Assuming input age is days? No, form usually years. 
    # WAIT, form input label says "Age" but previous code had label "Age (Years)" and logic `df['Age_Year'] = df['Age']`.
    # Let's assume input is YEARS (e.g. 50).
    df['Age_Year'] = df['Age'] 
    
    df['bili_umolL'] = df['Bilirubin'] * 17.1
    df['alb_gL'] = df['Albumin'] * 10
    df['plt_1000uL'] = df['Platelets'] / 1000
    df['ALBI'] = (np.log10(df['bili_umolL']) * 0.66) + (df['alb_gL'] * -0.085)
    df['PALBI'] = (df['ALBI'] * 1.0) + (df['plt_1000uL'] * -0.04)
    df['APRI'] = (df['SGOT'] / 40) / df['plt_1000uL']
    
    log_cols = ['Bilirubin', 'Copper', 'Alk_Phos', 'Tryglicerides', 'SGOT', 'Prothrombin', 'Cholesterol']
    for col in log_cols:
        df[col] = np.log1p(df[col])
        
    # Inference
    X_s3_scaled, _ = prepare_inference_data(sys_s3, df)
    prob_s3 = get_avg_prob(sys_s3, X_s3_scaled)[0]
    th_s3 = sys_s3['meta']['threshold']
    
    X_s2_scaled, _ = prepare_inference_data(sys_s2, df)
    prob_s2 = get_avg_prob(sys_s2, X_s2_scaled)[0]
    th_s2 = sys_s2['meta']['threshold']
    
    final_stage = 1
    shap_model_system = None
    shap_X = None
    shap_target_name = ""
    
    # Cascade Logic
    if prob_s3 > th_s3:
        final_stage = 3
        shap_model_system = sys_s3
        shap_X = X_s3_scaled
        shap_target_name = "Stage 3 (High Risk)"
    else:
        if prob_s2 > th_s2:
            final_stage = 2
            shap_model_system = sys_s2
            shap_X = X_s2_scaled
            shap_target_name = "Stage 2 vs 1"
        else:
            final_stage = 1
            shap_model_system = sys_s2
            shap_X = X_s2_scaled
            shap_target_name = "Stage 1 (Low Risk)"

    # Final Probabilities Estimation
    p3 = prob_s3
    p2 = (1 - p3) * prob_s2
    p1 = (1 - p3) * (1 - prob_s2)
    
    # SHAP Generation (LGBM Fixed)
    shap_image = None
    try:
        model = shap_model_system['models'].get('lgbm')
        if model:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(shap_X)
            
            # LGBM often returns list [class0, class1] for binary
            # We want positive class (index 1) explanation usually
            if isinstance(shap_values, list):
                shap_val = shap_values[1][0]
                base_val = explainer.expected_value[1]
            else:
                 # If binary, shap_values might be just 1 array?
                 if len(shap_values.shape) == 2:
                     shap_val = shap_values[0] # (1, features)
                     base_val = explainer.expected_value
                 else:
                     # 3D (1, feat, 2) ?
                     shap_val = shap_values[0, :, 1]
                     base_val = explainer.expected_value[1]
            
            plt.figure()
            shap.waterfall_plot(
                shap.Explanation(values=shap_val, base_values=base_val, 
                                 data=shap_X.iloc[0].values, 
                                 feature_names=shap_X.columns.tolist()),
                show=False, max_display=10
            )
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            shap_image = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
    except Exception as e:
        print(f"SHAP Error: {e}")

    result = {
        'final_stage': final_stage,
        'prob_s3': prob_s3,
        'th_s3': th_s3,
        'prob_s2': prob_s2,
        'th_s2': th_s2,
        'final_probs': {'p3': p3, 'p2': p2, 'p1': p1},
        'shap_target': shap_target_name
    }
    
    from jinja2 import Template
    t = Template(html_template)
    return HTMLResponse(content=t.render(result=result, shap_image=shap_image))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)