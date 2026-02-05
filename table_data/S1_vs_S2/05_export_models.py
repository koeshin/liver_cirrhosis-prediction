import pandas as pd
import numpy as np
import os
import json
import joblib
import argparse
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from logger_utils import LogTee

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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def get_scaler(name):
    if name == 'standard': return StandardScaler()
    if name == 'minmax': return MinMaxScaler()
    if name == 'robust': return RobustScaler()
    raise ValueError(f"Unknown scaler: {name}")

def load_params(dataset_name, scaler_name, model_name):
    path = f"artifacts/best_params/{dataset_name}/{scaler_name}/{model_name}.json"
    if not os.path.exists(path):
        print(f"Warning: Params not found at {path}")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def optimize_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_th = 0.5
    best_f1 = 0.0
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1

def generate_scores_feature(X, y_train, num_cols, top3_cols=None, weights=None):
    scaler_score = StandardScaler()
    
    if top3_cols is None or weights is None:
        print("Calculating Top 3 Score Weights from Full Train...")
        ref_model = LGBMClassifier(random_state=42, verbose=-1)
        ref_model.fit(X, y_train)
        
        importances = pd.Series(ref_model.feature_importances_, index=X.columns)
        num_imps = importances[importances.index.isin(num_cols)].sort_values(ascending=False)
        top3_cols = num_imps.head(3).index.tolist()
        top3_vals = num_imps.head(3).values
        weights = top3_vals / top3_vals.sum() if len(top3_vals) > 0 else []
        scaler_score.fit(X[top3_cols])
    else:
        scaler_score.fit(X[top3_cols]) 
        
    X_t3 = scaler_score.transform(X[top3_cols])
    score = np.dot(X_t3, weights)
    
    return score, top3_cols, weights, scaler_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='B')
    parser.add_argument('--scaler', type=str, default='robust')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    scaler_name = args.scaler
    
    log_path = f"artifacts/logs/export_{dataset_name}.log"
    out_dir = "artifacts/final_models"
    os.makedirs(out_dir, exist_ok=True)
    
    with LogTee(log_path):
        print(f"--- Exporting Models: Dataset {dataset_name} + {scaler_name} Scaler ---")
        
        train_path = f"data/processed/{dataset_name}.parquet"
        meta_path = 'data/processed/dataset_meta.json'
        
        if not os.path.exists(train_path):
            print("Train data missing")
            return
            
        train_df = pd.read_parquet(train_path)
        y = train_df['Stage']
        X = train_df.drop('Stage', axis=1)
        X = pd.get_dummies(X, drop_first=True)
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        num_cols = meta['num_cols'][dataset_name]
        
        print("Generating Ensemble Score Feature...")
        score_vals, top3, weights, score_scaler = generate_scores_feature(X, y, num_cols)
        X['top3_score'] = score_vals
        
        score_meta = {
            'top3_cols': top3,
            'weights': weights.tolist(),
            'score_scaler_mean': score_scaler.mean_.tolist(),
            'score_scaler_scale': score_scaler.scale_.tolist()
        }
        
        print(f"Applying {scaler_name} scaler...")
        main_scaler = get_scaler(scaler_name)
        feature_names = X.columns.tolist()
        X_scaled = pd.DataFrame(main_scaler.fit_transform(X), columns=feature_names)
        
        joblib.dump(main_scaler, f"{out_dir}/scaler.pkl")
        
        models = ['rf', 'xgb', 'lgbm']
        model_paths = {}
        oof_preds = {}
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("Training Models...")
        for m_name in models:
            print(f"  > {m_name}")
            params = load_params(dataset_name, scaler_name, m_name)
            if not params:
                print(f"Skipping {m_name} (no params)")
                continue
                
            if m_name == 'rf': clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
            elif m_name == 'xgb': clf = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
            else: clf = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, **params)
            
            clf.fit(X_scaled, y)
            joblib.dump(clf, f"{out_dir}/{m_name}.pkl")
            model_paths[m_name] = f"{out_dir}/{m_name}.pkl"
            
            oof_p = np.zeros(len(X))
            for tr_idx, val_idx in kf.split(X_scaled, y):
                f_X_tr = X_scaled.iloc[tr_idx]
                f_y_tr = y.iloc[tr_idx]
                f_X_va = X_scaled.iloc[val_idx]
                
                m_fold = None
                if m_name == 'rf': m_fold = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
                elif m_name == 'xgb': m_fold = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
                else: m_fold = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, **params)
                
                m_fold.fit(f_X_tr, f_y_tr)
                oof_p[val_idx] = m_fold.predict_proba(f_X_va)[:, 1]
            oof_preds[m_name] = oof_p

        print("Optimizing Ensemble Threshold...")
        if oof_preds:
            ens_oof = np.mean(list(oof_preds.values()), axis=0)
            best_th, best_f1 = optimize_threshold(y, ens_oof)
            auc = roc_auc_score(y, ens_oof)
            print(f"  Ensemble OOF AUC: {auc:.4f}, Best Th: {best_th:.2f}, F1: {best_f1:.4f}")
            
            final_meta = {
                'dataset': dataset_name,
                'scaler': scaler_name,
                'features': feature_names,
                'threshold': best_th,
                'score_meta': score_meta,
                'metrics': {'auc': auc, 'f1': best_f1}
            }
            with open(f"{out_dir}/model_meta.json", 'w') as f:
                json.dump(final_meta, f, indent=4)
            print(f"Artifacts saved to {out_dir}")
        else:
            print("No models trained!")

if __name__ == "__main__":
    main()
