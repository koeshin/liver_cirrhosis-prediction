import pandas as pd
import numpy as np
import os
import json
import optuna
import joblib
import argparse
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings

import logging
import sys

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# Configure Optuna to log to stdout (captured by LogTee)
optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.handlers = [] # Remove default handlers
optuna_logger.addHandler(logging.StreamHandler(sys.stdout))

def get_scaler(name):
    if name == 'standard': return StandardScaler()
    if name == 'minmax': return MinMaxScaler()
    if name == 'robust': return RobustScaler()
    raise ValueError(f"Unknown scaler: {name}")

def get_num_cols(df, meta, dataset_name):
    # Retrieve numerical columns from metadata or infer
    # Here using inference as backup
    return df.select_dtypes(include=[np.number]).columns.drop('Stage', errors='ignore').tolist()

def generate_in_fold_scores(X, y, num_cols, n_splits=5):
    """
    Generates 'Top3 Weighted Score' features in a leakage-free manner (inside CV).
    Returns: DataFrame with added 'top3_score' feature.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = np.zeros(len(X))
    
    print(f"Generating In-Fold Scores ({n_splits}-fold)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        # 1. Train Reference Model (LGBM) for Feature Importance
        # Numerical features only for importance scoring? 
        # User Plan: "Importance calculation -> Filter numeric -> Top 3"
        # We use all features to train, then look at importance for numerical ones.
        
        ref_model = LGBMClassifier(random_state=42, verbose=-1)
        ref_model.fit(X_train, y_train)
        
        importances = pd.Series(ref_model.feature_importances_, index=X_train.columns)
        
        # 2. Filter Numerical & Top 3
        # Ensure we only pick from 'num_cols'
        num_importances = importances[importances.index.isin(num_cols)].sort_values(ascending=False)
        top3_cols = num_importances.head(3).index.tolist()
        top3_vals = num_importances.head(3).values
        
        # 3. Normalize Weights
        if len(top3_vals) > 0:
            weights = top3_vals / top3_vals.sum()
            print(f"  [Fold {fold}] Top 3 Features: {top3_cols}, Weights: {weights}")
        else:
            weights = []
            
        # 4. Generate Score
        # Score = w1*f1 + w2*f2 + w3*f3
        # Note: Features should be standardized?
        # The plan says: "(Scaler applied) Model training" for step 1.
        # But for score generation itself step 4?
        # Usually weighted sum requires comparable scales.
        # We will apply Standard scaler to Top3 columns on X_train for weight calculation context?
        # No, LGBM is scale invariant. But the weighted sum implies linearity.
        # Plan says: "1. (Scaler applied) Model training".
        # So we should scale X_train before finding importance? LGBM doesn't care.
        # But for the *score* itself (weighted sum), scale matters if we just sum them.
        # We will Standard Scale the Top 3 features before summing.
        
        scaler_score = StandardScaler()
        X_train_top3 = scaler_score.fit_transform(X_train[top3_cols])
        X_val_top3 = scaler_score.transform(X_val[top3_cols])
        
        score_train = np.dot(X_train_top3, weights)
        score_val = np.dot(X_val_top3, weights)
        
        # Fill scores
        scores[val_idx] = score_val
        
        # For Optimization, we usually need 'train' scores too if we refit?
        # Here we only need 'scores' as a feature for the dataset.
        # But wait, this produces 'OOF' scores for the entire dataset used for Tuning.
        # Correct. The tuned models will see 'top3_score' as just another column.
        
    X_with_score = X.copy()
    X_with_score['top3_score'] = scores
    return X_with_score

def objective(trial, X, y, model_name):
    # Models
    if model_name == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif model_name == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(random_state=42, n_jobs=-1, **params)
    elif model_name == 'lgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        model = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, **params)
    
    # 5-fold CV for Objective
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Valid AUC
    aucs = []
    
    # To save time in `trial`, we can use `cross_val_score` but we want Stratified.
    # Also if scalers are involved, we might need Pipeline within CV if we were tuning scaler.
    # But here Scaler is fixed for this process loop. 'X' passed here is already scaled?
    # No, 'X' is passed raw from 'tune_model' function which handles scaling.
    # BUT wait, the 'top3_score' was generated on the WHOLE train set via CV.
    # Is it data leakage to standard scale the whole X before CV? 
    # Yes. Scaler should be fit on fold-train.
    # So we used Pipeline.
    
    # However, 'tune_model' below:
    # 1. Generate Score (OOF) -> This is safe.
    # 2. Loop Scaler -> Here we apply scaler to the entire X? NO.
    # We should apply scalar inside CV loop of Optuna.
    # But the user plan says: "Tuning Targets: ... Scaler: standard/minmax/robust respectively".
    # This implies Scaler Type is a hyperparameter or an outer loop.
    # "Outer loop" in Step 4 description: "Dataset: A/B/C each... Scaler: standard/minmax/robust each".
    # So we fix Scaler for a Study.
    
    # Inside CV of objective function, we must fit transform train, transform val.
    
    cv_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Fit model
        # Note: If X contains object columns, we need encoding.
        # But we assume numeric or LGBM handles it. RF/XGB might need encoding.
        # For simplicity and 'Execute' constraint, we assume OneHot or similar was done?
        # 01 script didn't OHE.
        # LGBM handles cat. XGB/RF need encoding.
        # We'll use simple encoding (get_dummies) or assume numeric for now to proceed.
        # Actually Model S3 binary notebook used OHE.
        # We will convert object to category for LGBM, OHE for others.
        # Handled in main preparation.
        
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        cv_scores.append(score)
        
    return np.mean(cv_scores)

from logger_utils import LogTee

def tune_and_save(dataset_name, df, num_cols):
    log_path = f"artifacts/logs/tuning_{dataset_name}.log"
    with LogTee(log_path):
        print(f"Logging entire session to {log_path}")
        
        X = df.drop('Stage', axis=1)
        y = df['Stage']
        
        # Handle Categoricals (Naive OHE for compatibility)
        X = pd.get_dummies(X, drop_first=True)
        
        # 1. Generate Score
        X_scored = generate_in_fold_scores(X, y, num_cols)
        
        # 2. Loops
        scalers = ['standard', 'minmax', 'robust']
        models = ['rf', 'xgb', 'lgbm']
        
        for s_name in scalers:
            print(f"\n--- Tuning Scaler: {s_name} ---")
            
            # Correct approach:
            scaler_cls = get_scaler(s_name)
        
            for m_name in models:
                print(f"Model: {m_name}")
                
                study_name = f"{dataset_name}_{s_name}_{m_name}"
                storage_path = f"sqlite:///artifacts/optuna_studies/{study_name}.db"
                
                # Helper for objective to use specific scaler/model
                def obj_wrapper(trial):
                    # Apply Scaler inside CV
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    scores = []
                    
                    # Model params selection
                    if m_name == 'rf':
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                            'max_depth': trial.suggest_int('max_depth', 3, 20),
                            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                        }
                        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
                    elif m_name == 'xgb':
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                            'max_depth': trial.suggest_int('max_depth', 3, 10),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                        }
                        model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
                    else: # lgbm
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                            'max_depth': trial.suggest_int('max_depth', 3, 15),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                        }
                        model = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, **params)

                    for train_idx, val_idx in skf.split(X_scored, y):
                        X_tr, y_tr = X_scored.iloc[train_idx], y.iloc[train_idx]
                        X_va, y_va = X_scored.iloc[val_idx], y.iloc[val_idx]
                        
                        # Scaling
                        s = scaler_cls
                        X_tr_s = s.fit_transform(X_tr)
                        X_va_s = s.transform(X_va)
                        
                        model.fit(X_tr_s, y_tr)
                        preds = model.predict_proba(X_va_s)[:, 1]
                        scores.append(roc_auc_score(y_va, preds))
                    
                    return np.mean(scores)

                study = optuna.create_study(direction='maximize', storage=storage_path, study_name=study_name, load_if_exists=True)
                study.optimize(obj_wrapper, n_trials=20) # 20 trials for demonstration/speed
                
                print(f"Best value: {study.best_value}")
                print(f"Best params: {study.best_params}")
                
                # Save best params
                out_dir = f"artifacts/best_params/{dataset_name}/{s_name}"
                os.makedirs(out_dir, exist_ok=True)
                with open(f"{out_dir}/{m_name}.json", 'w') as f:
                    json.dump(study.best_params, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['A', 'B', 'C'], required=True)
    args = parser.parse_args()
    
    data_path = f"data/processed/{args.dataset}.parquet"
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found.")
        return
        
    df = pd.read_parquet(data_path)
    
    # Load metadata for num columns
    with open('data/processed/dataset_meta.json', 'r') as f:
        meta = json.load(f)
        
    num_cols = meta['num_cols'][args.dataset]
    
    tune_and_save(args.dataset, df, num_cols)

if __name__ == "__main__":
    main()
