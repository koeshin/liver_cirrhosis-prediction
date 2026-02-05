import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
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

def get_scaler(name):
    if name == 'standard': return StandardScaler()
    if name == 'minmax': return MinMaxScaler()
    if name == 'robust': return RobustScaler()
    return StandardScaler() # Default

def load_best_params(dataset_name):
    root = f"artifacts/best_params/{dataset_name}"
    models_config = []
    
    if not os.path.exists(root):
        print(f"No params found for {dataset_name} in {root}")
        return []
        
    for s_name in os.listdir(root):
        s_path = os.path.join(root, s_name)
        if os.path.isdir(s_path):
            for m_file in os.listdir(s_path):
                if m_file.endswith('.json'):
                    m_name = m_file.replace('.json', '')
                    with open(os.path.join(s_path, m_file), 'r') as f:
                        params = json.load(f)
                    models_config.append({
                        'scaler': s_name,
                        'model': m_name,
                        'params': params
                    })
    return models_config

def train_predict(X_train, y_train, X_test, config):
    # Scale
    scaler = get_scaler(config['scaler'])
    
    # Keep as DataFrame to avoid "valid feature names" warning in LGBM
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Model
    m_name = config['model']
    params = config['params']
    
    if m_name == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif m_name == 'xgb':
        model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
    else:
        model = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, **params)
        
    model.fit(X_tr_s, y_train)
    preds = model.predict_proba(X_te_s)[:, 1]
    return preds

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['A', 'B', 'C'], required=True, help='Dataset to use (A, B, or C)')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    
    log_path = f"artifacts/logs/ensemble_{dataset_name}.log"
    with LogTee(log_path):
        print(f"Starting Ensemble & Threshold Optimization for Dataset {dataset_name}")
        print(f"Logging to {log_path}")
        
        train_path = f"data/processed/{dataset_name}.parquet"
        test_path = "data/processed/test.parquet"
        meta_path = 'data/processed/dataset_meta.json'
        
        if not os.path.exists(train_path) or not os.path.exists(test_path) or not os.path.exists(meta_path):
            print(f"Data files missing. Ensure 01_build_datasets.py ran successfully.")
            return

        train_df = pd.read_parquet(train_path)
        test_df_raw = pd.read_parquet(test_path)
        
        # Re-apply transformations to Test set
        log_cols = ['Bilirubin', 'Copper', 'Alk_Phos', 'Tryglicerides', 'SGOT', 'Prothrombin', 'Cholesterol']
        for col in log_cols:
            if col in test_df_raw.columns:
                test_df_raw[col] = np.log1p(test_df_raw[col])
                
        test_df_fe = medical_feature_engineering(test_df_raw)
        
        # 3. Drops (Global) matching 01 script
        global_drop_cols = [
            'N_Days', 'Status', 'Drug', 'Age', 'Sex',
            'Age_Year', 'bili_umolL', 'alb_gL', 'plt_1000uL'
        ]
        test_df_fe = test_df_fe.drop(columns=[c for c in global_drop_cols if c in test_df_fe.columns], errors='ignore')

        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        cols_needed = meta['features'][dataset_name]
        num_cols = meta['num_cols'][dataset_name]
        
        print(f"Columns selected for Dataset {dataset_name} ({len(cols_needed)} cols): {cols_needed}")
        
        X_train = train_df.drop('Stage', axis=1)
        y_train = train_df['Stage']
        
        # Select columns for Test
        X_test = test_df_fe[cols_needed].copy()
        y_test = test_df_fe['Stage']
        
        # Handle Categoricals
        X_train = pd.get_dummies(X_train, drop_first=True)
        X_test = pd.get_dummies(X_test, drop_first=True)
        
        # Align
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        
        # Add Score Feature
        print("Generating scores for Ensemble...")
        ref_model = LGBMClassifier(random_state=42, verbose=-1)
        ref_model.fit(X_train, y_train)
        
        importances = pd.Series(ref_model.feature_importances_, index=X_train.columns)
        num_imps = importances[importances.index.isin(num_cols)].sort_values(ascending=False)
        top3_cols = num_imps.head(3).index.tolist()
        top3_vals = num_imps.head(3).values
        weights = top3_vals / top3_vals.sum() if len(top3_vals)>0 else []
        
        print(f"Top 3 Features selected for Ensemble Score: {top3_cols}")
        print(f"Weights: {weights}")
        
        sc = StandardScaler()
        # Ensure DF for consistency
        X_tr_t3_np = sc.fit_transform(X_train[top3_cols])
        X_te_t3_np = sc.transform(X_test[top3_cols])
        
        score_tr = np.dot(X_tr_t3_np, weights)
        score_te = np.dot(X_te_t3_np, weights)
        
        X_train['top3_score'] = score_tr
        X_test['top3_score'] = score_te
        
        # Load Configs
        configs = load_best_params(dataset_name)
        if not configs:
            print(f"No tuned models found for dataset {dataset_name}. Skipping Ensemble.")
            return

        print(f"\n--- 1. Evaluating All Individual Models ({len(configs)}) ---")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Store results
        results = []
        
        for cfg in configs:
            model_name = cfg['model']
            scaler_name = cfg['scaler']
            
            # 5-Fold CV for OOF
            oof_probs = np.zeros(len(X_train))
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr_fold, y_tr_fold = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val_fold = X_train.iloc[val_idx]
                p_val = train_predict(X_tr_fold, y_tr_fold, X_val_fold, cfg)
                oof_probs[val_idx] = p_val
                
            # Optimize Threshold
            best_th, best_f1 = optimize_threshold(y_train, oof_probs)
            auc = roc_auc_score(y_train, oof_probs)
            
            print(f"Model: {model_name: <5} | Scaler: {scaler_name: <10} | OOF AUC: {auc:.4f} | F1: {best_f1:.4f} | Th: {best_th:.2f}")
            
            results.append({
                'config': cfg,
                'model': model_name,
                'scaler': scaler_name,
                'oof_probs': oof_probs,
                'best_th': best_th,
                'best_f1': best_f1,
                'auc': auc
            })

        print("\n--- 2. Ensembling by Scaler Group ---")
        
        scalers = ['standard', 'minmax', 'robust']
        ensemble_results = []
        
        for s_name in scalers:
            # Get models for this scaler
            group_results = [r for r in results if r['scaler'] == s_name]
            
            if not group_results:
                print(f"No models found for {s_name} scaler.")
                continue
                
            print(f"\n[Ensemble: {s_name.upper()} Scaler Group]")
            print(f"  Combining models: {[r['model'] for r in group_results]}")
            
            # Simple Average of OOF probs (Soft Voting)
            ens_oof_probs = np.mean([r['oof_probs'] for r in group_results], axis=0)
            
            # Optimize Threshold for Ensemble
            ens_th, ens_f1 = optimize_threshold(y_train, ens_oof_probs)
            ens_auc = roc_auc_score(y_train, ens_oof_probs)
            
            print(f"  -> Ensemble OOF AUC: {ens_auc:.4f}")
            print(f"  -> Ensemble OOF F1 : {ens_f1:.4f} (at Th={ens_th:.2f})")
            
            # Evaluate on Test Set
            print(f"  Evaluating on Test Set...")
            ens_test_probs = np.zeros(len(X_test))
            
            # Retrain all models in group on full train and predict test
            for res in group_results:
                cfg = res['config']
                p_test = train_predict(X_train, y_train, X_test, cfg)
                ens_test_probs += p_test
            
            ens_test_probs /= len(group_results)
            
            # Final Metris
            test_auc = roc_auc_score(y_test, ens_test_probs)
            pred_labels = (ens_test_probs >= ens_th).astype(int)
            test_f1 = f1_score(y_test, pred_labels)
            test_acc = accuracy_score(y_test, pred_labels)
            
            print(f"  -> TEST AUC:      {test_auc:.4f}")
            print(f"  -> TEST F1:       {test_f1:.4f}")
            print(f"  -> TEST Accuracy: {test_acc:.4f}")
            print(classification_report(y_test, pred_labels))
            
            ensemble_results.append({
                'scaler': s_name,
                'auc': test_auc,
                'f1': test_f1,
                'acc': test_acc,
                'th': ens_th
            })

        print("\n--- 3. Best Ensemble Selection ---")
        if ensemble_results:
            best_ens = sorted(ensemble_results, key=lambda x: x['f1'], reverse=True)[0]
            print(f"\n🏆 Best Ensemble: {best_ens['scaler'].upper()} Scaler Group")
            print(f"   AUC: {best_ens['auc']:.4f}")
            print(f"   F1 : {best_ens['f1']:.4f}")
            print(f"   Acc: {best_ens['acc']:.4f}")
            print(f"   Th : {best_ens['th']:.2f}")
        else:
            print("No ensembles created.")

if __name__ == "__main__":
    main()
