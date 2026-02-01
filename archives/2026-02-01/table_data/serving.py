
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

# Initialize FastAPI
app = FastAPI(title="Liver Cirrhosis Progression Prediction")

# Load Models (Global variables)
models = {}
preprocessor = None
best_params = {}

def load_models():
    global models, preprocessor, best_params
    model_dir = "saved_models"
    
    print("Loading models...")
    try:
        models['Random Forest'] = joblib.load(os.path.join(model_dir, "model_b_random_forest.pkl"))
        models['XGBoost'] = joblib.load(os.path.join(model_dir, "model_b_xgboost.pkl"))
        models['LightGBM'] = joblib.load(os.path.join(model_dir, "model_b_lightgbm.pkl"))
        models['Ensemble'] = joblib.load(os.path.join(model_dir, "model_b_voting_ensemble.pkl"))
        
        preprocessor = joblib.load(os.path.join(model_dir, "model_b_preprocessor.pkl"))
        
        with open(os.path.join(model_dir, "model_b_best_params.json"), 'r') as f:
            best_params = json.load(f)
            
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# Feature Engineering Function (Must match notebook logic)
def engineer_features(data: dict) -> pd.DataFrame:
    # Create DataFrame from single input
    df = pd.DataFrame([data])
    
    # 1. Medical Feature Engineering
    float_cols = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Age: Input is now in Years
    df['Age_Year'] = df['Age']
    df['Age'] = df['Age'] * 365.25 # Keep days for consistency if needed elsewhere
    
    # Unit conversions
    df['bili_umolL'] = df['Bilirubin'] * 17.1
    df['alb_gL'] = df['Albumin'] * 10
    df['plt_1000uL'] = df['Platelets'] / 1000
    
    # ALBI Score
    df['ALBI'] = (np.log10(df['bili_umolL']) * 0.66) + (df['alb_gL'] * -0.085)
    
    # PALBI Score  
    df['PALBI'] = (df['ALBI'] * 1.0) + (df['plt_1000uL'] * -0.04)
    
    # APRI
    df['APRI'] = (df['SGOT'] / 40) / df['plt_1000uL']
    
    # FIB-4
    df['FIB4'] = (df['Age_Year'] * df['SGOT']) / (df['plt_1000uL'] * np.sqrt(df['Bilirubin']))
    
    # Bilirubin/Platelets
    df['Bili_Platelet_Ratio'] = df['Bilirubin'] / (df['Platelets'] + 1)
    
    # Copper×Bilirubin
    df['Copper_Bili_Interaction'] = df['Copper'] * df['Bilirubin']
    
    # 2. Log Transformation
    log_cols = ['Bilirubin', 'Copper', 'Alk_Phos', 'Tryglicerides', 'SGOT', 'Prothrombin', 'Cholesterol']
    for col in log_cols:
         df[col] = np.log1p(df[col])
    
    return df

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Liver Cirrhosis Stage Prediction (Model B)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f8f9fa; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .result-box { padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .stage-1 { background-color: #d4edda; color: #155724; }
        .stage-2 { background-color: #fff3cd; color: #856404; }
        .stage-3 { background-color: #f8d7da; color: #721c24; }
        .feature-group { background-color: #fff; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">🩺 Liver Cirrhosis Progression Prediction</h1>
        <p class="text-center text-muted">Predicting Stage 1 vs 2 vs 3 (Model B)</p>
        
        <div class="row">
            <div class="col-md-5">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">Patient Data</h4>
                    </div>
                    <div class="card-body">
                        <form action="/predict" method="post">
                            <div class="feature-group">
                                <h5>Basic Info</h5>
                                <div class="mb-2">
                                    <label>Age (Years)</label>
                                    <input type="number" name="Age" class="form-control" value="50" required>
                                </div>
                                <div class="mb-2">
                                    <label>Sex</label>
                                    <select name="Sex" class="form-select">
                                        <option value="F">Female</option>
                                        <option value="M">Male</option>
                                    </select>
                                </div>
                            </div>

                            <div class="feature-group">
                                <h5>Blood Tests</h5>
                                <div class="row">
                                    <div class="col-6 mb-2">
                                        <label>Bilirubin (mg/dL)</label>
                                        <input type="number" step="0.1" name="Bilirubin" class="form-control" value="1.0" required>
                                    </div>
                                    <div class="col-6 mb-2">
                                        <label>Albumin (g/dL)</label>
                                        <input type="number" step="0.1" name="Albumin" class="form-control" value="3.5" required>
                                    </div>
                                    <div class="col-6 mb-2">
                                        <label>Copper (ug/day)</label>
                                        <input type="number" step="1" name="Copper" class="form-control" value="50" required>
                                    </div>
                                    <div class="col-6 mb-2">
                                        <label>Alk_Phos (U/L)</label>
                                        <input type="number" step="1" name="Alk_Phos" class="form-control" value="1000" required>
                                    </div>
                                    <div class="col-6 mb-2">
                                        <label>SGOT (U/mL)</label>
                                        <input type="number" step="0.1" name="SGOT" class="form-control" value="100" required>
                                    </div>
                                    <div class="col-6 mb-2">
                                        <label>Cholesterol (mg/dL)</label>
                                        <input type="number" step="1" name="Cholesterol" class="form-control" value="300" required>
                                    </div>
                                     <div class="col-6 mb-2">
                                        <label>Tryglicerides (mg/dL)</label>
                                        <input type="number" step="1" name="Tryglicerides" class="form-control" value="100" required>
                                    </div>
                                    <div class="col-6 mb-2">
                                        <label>Platelets (1000/uL)</label>
                                        <input type="number" step="1" name="Platelets" class="form-control" value="250" required>
                                    </div>
                                    <div class="col-6 mb-2">
                                        <label>Prothrombin (s)</label>
                                        <input type="number" step="0.1" name="Prothrombin" class="form-control" value="10.0" required>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="feature-group">
                                <h5>Physical Symptoms (Y/N)</h5>
                                <div class="row">
                                    <div class="col-4 mb-2">
                                        <label>Ascites</label>
                                        <select name="Ascites" class="form-select">
                                            <option value="N">No</option>
                                            <option value="Y">Yes</option>
                                        </select>
                                    </div>
                                    <div class="col-4 mb-2">
                                        <label>Hepatomegaly</label>
                                        <select name="Hepatomegaly" class="form-select">
                                            <option value="N">No</option>
                                            <option value="Y">Yes</option>
                                        </select>
                                    </div>
                                    <div class="col-4 mb-2">
                                        <label>Spiders</label>
                                        <select name="Spiders" class="form-select">
                                            <option value="N">No</option>
                                            <option value="Y">Yes</option>
                                        </select>
                                    </div>
                                    <div class="col-6 mb-2">
                                        <label>Edema</label>
                                        <select name="Edema" class="form-select">
                                            <option value="N">No Edema</option>
                                            <option value="S">Edema (No Diuretics)</option>
                                            <option value="Y">Edema (With Diuretics)</option>
                                        </select>
                                    </div>
                                </div>
                            </div>

                            <button type="submit" class="btn btn-primary w-100 btn-lg">Predict Stage</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-7">
                {% if prediction_results %}
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h4 class="mb-0">Prediction Results</h4>
                    </div>
                    <div class="card-body">
                        <h5>Model Consensus</h5>
                        <div class="row">
                            {% for model_name, result in prediction_results.items() %}
                            <div class="col-md-6">
                                <div class="result-box border">
                                    <strong>{{ model_name }}</strong>
                                    <div class="d-flex justify-content-between align-items-center mt-2">
                                        <span class="badge bg-secondary">Stage {{ result.stage }}</span>
                                        <small>Confidence: {{ "%.1f"|format(result.confidence * 100) }}%</small>
                                    </div>
                                    <div class="progress mt-2" style="height: 5px;">
                                        <div class="progress-bar" role="progressbar" style="width: {{ result.confidence * 100 }}%"></div>
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header bg-info text-dark">
                        <h4 class="mb-0">🔍 Why this prediction? (SHAP Analysis)</h4>
                    </div>
                    <div class="card-body text-center">
                        <p class="text-muted">Analysis based on <strong>{{ best_model_name }}</strong> (Highest Confidence)</p>
                        {% if shap_image %}
                        <img src="data:image/png;base64,{{ shap_image }}" class="img-fluid" alt="SHAP Waterfall Plot">
                        {% else %}
                        <p>SHAP visualization failed to generate.</p>
                        {% endif %}
                    </div>
                </div>
                
                <div class="card">
                     <div class="card-header bg-light">
                        <h5 class="mb-0">Derived Medical Features</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                             <div class="col-md-4 mb-2"><strong>ALBI:</strong> {{ "%.2f"|format(features.ALBI) }}</div>
                             <div class="col-md-4 mb-2"><strong>APRI:</strong> {{ "%.2f"|format(features.APRI) }}</div>
                             <div class="col-md-4 mb-2"><strong>FIB-4:</strong> {{ "%.2f"|format(features.FIB4) }}</div>
                             <div class="col-md-4 mb-2"><strong>PALBI:</strong> {{ "%.2f"|format(features.PALBI) }}</div>
                             <div class="col-md-8"><strong>Bili/Platelet:</strong> {{ "%.4f"|format(features.Bili_Platelet_Ratio) }}</div>
                        </div>
                    </div>
                </div>
                {% else %}
                <div class="alert alert-info">
                    👈 Enter patient data on the left to generate predictions.
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
    return t.render(prediction_results=None)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  Age: float = Form(...), Sex: str = Form(...), 
                  Bilirubin: float = Form(...), Albumin: float = Form(...),
                  Copper: float = Form(...), Alk_Phos: float = Form(...),
                  SGOT: float = Form(...), Cholesterol: float = Form(...),
                  Tryglicerides: float = Form(...), Platelets: float = Form(...),
                  Prothrombin: float = Form(...), 
                  Ascites: str = Form(...), Hepatomegaly: str = Form(...),
                  Spiders: str = Form(...), Edema: str = Form(...)):
    
    # 1. Prepare Data
    input_data = {
        'Age': Age, 'Sex': Sex, 'Bilirubin': Bilirubin, 'Albumin': Albumin,
        'Copper': Copper, 'Alk_Phos': Alk_Phos, 'SGOT': SGOT, 
        'Cholesterol': Cholesterol, 'Tryglicerides': Tryglicerides, 
        'Platelets': Platelets, 'Prothrombin': Prothrombin,
        'Ascites': Ascites, 'Hepatomegaly': Hepatomegaly, 
        'Spiders': Spiders, 'Edema': Edema
    }
    
    # 2. Engineer Features
    df_features = engineer_features(input_data)
    
    inference_features = [
        'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 
        'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 
        'Tryglicerides', 'Platelets', 'Prothrombin', 
        'ALBI', 'PALBI', 'APRI', 'FIB4', 'Bili_Platelet_Ratio', 'Copper_Bili_Interaction'
    ]
    
    df_inference = df_features[inference_features].copy()
    
    # 3. Model Inference & results
    prediction_results = {}
    best_confidence = -1
    best_model_name = ""
    
    for name, model in models.items():
        try:
            pred_class = model.predict(df_inference)[0] # 0, 1, 2
            pred_proba = model.predict_proba(df_inference)[0]
            confidence = float(pred_proba[pred_class])
            
            prediction_results[name] = {
                'stage': int(pred_class + 1),
                'confidence': confidence
            }
            
            # Find best model based on confidence (excluding Ensemble for SHAP if possible, or just the best)
            if name != 'Ensemble' and confidence > best_confidence:
                best_confidence = confidence
                best_model_name = name
                
        except Exception as e:
            print(f"Error predicting with {name}: {e}")
            prediction_results[name] = {'stage': 'Error', 'confidence': 0.0}

    # Backup if all failed or only ensemble exists
    if not best_model_name:
         best_model_name = 'LightGBM'

    # 4. SHAP Visualization (Best Confidence Model)
    shap_image = None
    try:
        model_to_explain = models.get(best_model_name)
        if model_to_explain:
            pipeline_steps = dict(model_to_explain.steps)
            preproc = pipeline_steps['preprocessor']
            classifier = pipeline_steps['classifier']
            
            X_transformed = preproc.transform(df_inference)
            
            feature_names = []
            for name, transformer, features in preproc.transformers_:
                if name == 'num':
                    feature_names.extend(features)
                elif name == 'cat':
                    if hasattr(transformer, 'get_feature_names_out'):
                        feature_names.extend(transformer.get_feature_names_out(features))
            
            # Use TreeExplainer for XGB, LGBM, RF
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            
            pred_idx = prediction_results[best_model_name]['stage'] - 1
            
            # Multi-class output varies by library
            if isinstance(shap_values, list): # RF, sometimes LGBM
                shap_val_sample = shap_values[pred_idx][0]
                expected_val = explainer.expected_value[pred_idx]
            else: # XGB or newer LGBM
                if len(shap_values.shape) == 3: # (samples, features, classes)
                    shap_val_sample = shap_values[0, :, pred_idx]
                    expected_val = explainer.expected_value[pred_idx]
                else: # Binary or flattened?
                    shap_val_sample = shap_values[0]
                    expected_val = explainer.expected_value
            
            plt.figure()
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_val_sample,
                    base_values=expected_val,
                    data=X_transformed[0],
                    feature_names=feature_names
                ),
                show=False,
                max_display=12
            )
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            shap_image = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
    except Exception as e:
        print(f"SHAP generation error for {best_model_name}: {e}")
        import traceback
        traceback.print_exc()

    # Render Template
    from jinja2 import Template
    t = Template(html_template)
    html_content = t.render(
        prediction_results=prediction_results,
        shap_image=shap_image,
        features=df_features.iloc[0].to_dict(),
        best_model_name=best_model_name
    )
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)