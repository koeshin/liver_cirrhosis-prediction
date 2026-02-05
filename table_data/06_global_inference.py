import pandas as pd
import numpy as np
import os
import json
import joblib
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from logger_utils import LogTee

# --- Feature Engineering Functions (Must match training exactly) ---
def medical_feature_engineering(df_input):
    df_eng = df_input.copy()
    # Age conversion
    if 'Age' in df_eng.columns:
        df_eng['Age_Year'] = df_eng['Age'] / 365.25
    # Unit conversions
    if 'Bilirubin' in df_eng.columns:
        df_eng['bili_umolL'] = df_eng['Bilirubin'] * 17.1
    if 'Albumin' in df_eng.columns:
        df_eng['alb_gL'] = df_eng['Albumin'] * 10
    if 'Platelets' in df_eng.columns:
        df_eng['plt_1000uL'] = df_eng['Platelets'] / 1000
    
    # Scores
    if 'bili_umolL' in df_eng.columns and 'alb_gL' in df_eng.columns:
        df_eng['ALBI'] = (np.log10(df_eng['bili_umolL']) * 0.66) + (df_eng['alb_gL'] * -0.085)
    
    if 'ALBI' in df_eng.columns and 'plt_1000uL' in df_eng.columns:
        df_eng['PALBI'] = (df_eng['ALBI'] * 1.0) + (df_eng['plt_1000uL'] * -0.04)
    
    if 'SGOT' in df_eng.columns and 'plt_1000uL' in df_eng.columns:
        df_eng['APRI'] = (df_eng['SGOT'] / 40) / df_eng['plt_1000uL']
    
    return df_eng

def create_score_feature(X, meta, score_meta):
    # Recreate the score feature using saved metadata
    weights = np.array(score_meta['weights'])
    top3_cols = score_meta['top3_cols']
    
    if len(weights) == 0:
        return np.zeros(len(X))
        
    score_mean = np.array(score_meta['score_scaler_mean'])
    score_scale = np.array(score_meta['score_scaler_scale'])
    
    # 1. Select
    X_sub = X[top3_cols].values
    
    # 2. Scale (Standardize using saved mean/scale)
    X_sub_s = (X_sub - score_mean) / score_scale
    
    # 3. Dot Product
    scores = np.dot(X_sub_s, weights)
    return scores

def load_system(base_path):
    out_dir = f"{base_path}/artifacts/final_models"
    
    if not os.path.exists(f"{out_dir}/model_meta.json"):
        raise FileNotFoundError(f"Model artifacts not found in {out_dir}")
        
    with open(f"{out_dir}/model_meta.json", 'r') as f:
        meta = json.load(f)
        
    scaler = joblib.load(f"{out_dir}/scaler.pkl")
    
    models = []
    for m_name in ['rf', 'xgb', 'lgbm']:
        p = f"{out_dir}/{m_name}.pkl"
        if os.path.exists(p):
            models.append(joblib.load(p))
            
    return {
        'meta': meta,
        'scaler': scaler,
        'models': models
    }

def inference_step(system, X_df):
    meta = system['meta']
    scaler = system['scaler']
    models = system['models']
    
    # 1. Feature Prep
    # Ensure columns match training
    req_cols = meta['features'] # These are the scaled feature names (minus score?)
    # Wait, the Meta 'features' list from export script includes 'top3_score'.
    
    # Create Score Feature
    score_vals = create_score_feature(X_df, meta, meta['score_meta'])
    X_df_scored = X_df.copy()
    X_df_scored['top3_score'] = score_vals
    
    # One Hot (Naive matching)
    # Ideally we should align with saved feature list columns
    X_ohe = pd.get_dummies(X_df_scored, drop_first=True)
    
    # Reindex to match trained scaler expectation
    # The 'features' list in meta is exactly what went into scaler.fit
    final_cols = meta['features']
    
    # Add missing cols as 0
    for c in final_cols:
        if c not in X_ohe.columns:
            X_ohe[c] = 0
            
    # Select exact order
    X_ready = X_ohe[final_cols]
    
    # 2. Scale
    X_scaled = pd.DataFrame(scaler.transform(X_ready), columns=final_cols)
    
    # 3. Predict (Average Prob)
    probs = np.zeros(len(X_scaled))
    for m in models:
        probs += m.predict_proba(X_scaled)[:, 1]
    probs /= len(models)
    
    return probs

def main():
    log_path = "./global_inference.log"
    with LogTee(log_path):
        print("--- Global Hierarchical Inference Pipeline ---")
        
        # 1. Reconstruct Test Set with Original Labels (1, 2, 3)
        print("Reconstructing Test Set from Raw Data...")
        raw_path = "data/liver_cirrhosis_deduped.csv"
        # Since we are in root or sibling, check path relative to this script location
        # Assume script is running from 'table_data/' root if we moved it, 
        # OR running from S12_vs_S3.
        # Let's assume we run from 'C:/Users/Islab/Desktop/vibe_code/liver_cirrhosis-prediction/table_data' directory?
        # User prompt implies: "root table_data" or something.
        # Adjust paths:
        # If running from 'table_data':
        #   S12_vs_S3 is './S12_vs_S3'
        #   S1_vs_S2 is './S1_vs_S2'
        #   Data is '../data/liver...' (relative to S12 folder, so relative to root it is '../data/..')?
        #   Wait, '01' script in 'S12' used '../data/liver_cirrhosis_deduped.csv'.
        #   So 'table_data' is parent of 'S12'.
        #   So raw data is in 'table_data/../data/' -> 'data/'.
        #   Let's just use absolute path or try relative.
        
        raw_path = "./data/liver_cirrhosis_deduped.csv"

        print(f"Loading Raw: {raw_path}")
        df = pd.read_csv(raw_path)
        if 'ID' in df.columns: df = df.drop('ID', axis=1)
        df = df.dropna(subset=['Stage'])
        df_target = df[df['Stage'].isin([1, 2, 3])]
        
        # Split (Same seed 42)
        X = df_target.drop('Stage', axis=1)
        y = df_target['Stage']
        
        # We need the TEST portion
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        print(f"Test Set Size: {len(X_test)}")
        print(f"Test Class Dist: {y_test.value_counts().to_dict()}")
        
        # Preprocessing (Log Transform + FE)
        # Apply same FE as Training
        log_cols = ['Bilirubin', 'Copper', 'Alk_Phos', 'Tryglicerides', 'SGOT', 'Prothrombin', 'Cholesterol']
        X_test_proc = X_test.copy()
        for col in log_cols:
            if col in X_test_proc.columns:
                X_test_proc[col] = np.log1p(X_test_proc[col])
                
        X_test_fe = medical_feature_engineering(X_test_proc)
        
        # Load Systems
        print("\nLoading Models...")
        sys_s3 = load_system("S12_vs_S3")
        sys_s2 = load_system("S1_vs_S2")
        
        th_s3 = sys_s3['meta']['threshold']
        th_s2 = sys_s2['meta']['threshold']
        
        print(f"S12 vs S3 Threshold: {th_s3:.4f}")
        print(f"S1 vs S2 Threshold: {th_s2:.4f}")
        
        # --- Inference Cascade ---
        print("\nRunning Cascade Inference...")
        
        # Step 1: Predict S3 Prob
        probs_s3 = inference_step(sys_s3, X_test_fe)
        
        final_preds = []
        
        # Vectorized logic is tricky if Step 2 depends on filtering.
        # But we can calculate Step 2 probs for EVERYONE (ignoring cost) or just subset.
        # For simplicity/vectorization, calculate Step 2 for everyone.
        
        probs_s2 = inference_step(sys_s2, X_test_fe)
        
        final_stages = []
        
        for p3, p2 in zip(probs_s3, probs_s2):
            if p3 > th_s3:
                # Predicted Stage 3 (Class 1 of Model A)
                final_stages.append(3)
            else:
                # Predicted Stage 1 or 2 (Class 0 of Model A)
                # Now check Model B (Class 1 = Stage 2, Class 0 = Stage 1)
                # Ensure Model B targets were 0=S1, 1=S2. 
                # (Verified in Task Boundary: "Stage 1 -> 0, Stage 2 -> 1")
                if p2 > th_s2:
                    final_stages.append(2)
                else:
                    final_stages.append(1)
                    
        # Evaluate
        final_stages = np.array(final_stages)
        acc = accuracy_score(y_test, final_stages)
        macro_f1 = f1_score(y_test, final_stages, average='macro')
        
        print(f"\nObtained Accuracy: {acc:.4f}")
        print(f"Obtained Macro F1: {macro_f1:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, final_stages))
        print("\nClassification Report:")
        print(classification_report(y_test, final_stages))

if __name__ == "__main__":
    main()
