import pandas as pd
import numpy as np
import os
import json
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
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
    
    # Scores (Re-calculate if deleted, but likely present in A)
    if 'bili_umolL' in df_eng.columns and 'alb_gL' in df_eng.columns:
        df_eng['ALBI'] = (np.log10(df_eng['bili_umolL']) * 0.66) + (df_eng['alb_gL'] * -0.085)
    
    if 'ALBI' in df_eng.columns and 'plt_1000uL' in df_eng.columns:
        df_eng['PALBI'] = (df_eng['ALBI'] * 1.0) + (df_eng['plt_1000uL'] * -0.04)
    
    if 'SGOT' in df_eng.columns and 'plt_1000uL' in df_eng.columns:
        df_eng['APRI'] = (df_eng['SGOT'] / 40) / df_eng['plt_1000uL']
    
    return df_eng

def create_interaction_features(df):
    """
    Explicitly create interactions for top medical features.
    """
    df = df.copy()
    
    # Bilirubin * Albumin (Classic Liver function interaction)
    if 'Bilirubin' in df.columns and 'Albumin' in df.columns:
        df['inter_Bili_Alb'] = df['Bilirubin'] * df['Albumin']  # Raw interaction
        df['ratio_Bili_Alb'] = df['Bilirubin'] / (df['Albumin'] + 1e-6)
        
    # Copper * Prothrombin (Another strong pair)
    if 'Copper' in df.columns and 'Prothrombin' in df.columns:
        df['inter_Copp_Pro'] = df['Copper'] * df['Prothrombin']
        
    # Platelets vs SGOT (APRI components, but raw interaction)
    if 'Platelets' in df.columns and 'SGOT' in df.columns:
        df['inter_Plt_SGOT'] = df['Platelets'] * df['SGOT']
        
    return df

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
    parser.add_argument('--dataset', type=str, default='A', help='Dataset to use (Default: A)')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    n_trials = args.trials
    
    log_path = f"artifacts/logs/extreme_lgbm_{dataset_name}.log"
    with LogTee(log_path):
        print(f"Starting Extreme LGBM Optimization on Dataset {dataset_name}")
        print(f"Logging to {log_path}")
        
        # Paths
        train_path = f"data/processed/{dataset_name}.parquet"
        test_path = "data/processed/test.parquet"
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("Data files missing.")
            return

        # 1. Load Data
        train_df = pd.read_parquet(train_path)
        test_df_raw = pd.read_parquet(test_path)
        
        X_train = train_df.drop('Stage', axis=1)
        y_train = train_df['Stage']
        
        # 2. Preprocess Test (Same as Train)
        # Log Transform
        log_cols = ['Bilirubin', 'Copper', 'Alk_Phos', 'Tryglicerides', 'SGOT', 'Prothrombin', 'Cholesterol']
        for col in log_cols:
            if col in test_df_raw.columns:
                test_df_raw[col] = np.log1p(test_df_raw[col])
        
        # Medical FE
        test_df_fe = medical_feature_engineering(test_df_raw)
        
        # Global Drops
        global_drop_cols = ['N_Days', 'Status', 'Drug', 'Age', 'Sex', 'Age_Year', 'bili_umolL', 'alb_gL', 'plt_1000uL']
        test_df_fe = test_df_fe.drop(columns=[c for c in global_drop_cols if c in test_df_fe.columns], errors='ignore')
        
        # Select Columns matches Dataset A (or requested)
        cols_needed = X_train.columns.tolist()
        
        # 3. Add Interaction Features (New Step)
        print("Adding Interaction Features...")
        X_train = create_interaction_features(X_train)
        
        # For Test, we need to ensure we have base columns first
        X_test = test_df_fe.drop('Stage', axis=1) # Start with all FE
        # If dataset was restricted (B or C), we need to handle that, but typically A has everything.
        # Let's assume input dataset A already has selection.
        # But create_interaction_features needs inputs.
        X_test = create_interaction_features(X_test)
        
        # Align Test to Train
        X_test = X_test[X_train.columns] # Ensure strict matching
        y_test = test_df_fe['Stage']
        
        # 4. Native Categoricals
        cat_cols = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
        # Verify existence
        cat_features = [c for c in cat_cols if c in X_train.columns]
        print(f"Native Categorical Features: {cat_features}")
        
        # Convert to category type
        for c in cat_features:
            X_train[c] = X_train[c].astype('category')
            X_test[c] = X_test[c].astype('category')
            
        # 5. Deep Optuna Tuning
        print(f"\n--- Starting Deep Optuna Tuning ({n_trials} trials) ---")
        
        def objective(trial):
            param = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                # Categorical Params
                'cat_smooth': trial.suggest_int('cat_smooth', 1, 100),
                'cat_l2': trial.suggest_float('cat_l2', 1e-3, 50.0, log=True)
            }
            
            # 5-Fold Stratified CV
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            auc_scores = []
            
            for tr_idx, val_idx in skf.split(X_train, y_train):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
                
                # LGBM Dataset
                dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
                dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_features, reference=dtrain)
                
                # Manual Training call for Native Cat support in CV loop? 
                # Or just use sklearn API? 
                # Using sklearn API is easier for consistency, but passing cat_feature is cleaner in fit.
                
                model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, **param)
                model.fit(X_tr, y_tr, categorical_feature=cat_features) # Pass explicitly here
                preds = model.predict_proba(X_va)[:, 1]
                auc_scores.append(roc_auc_score(y_va, preds))
                
            return np.mean(auc_scores)

        study = optuna.create_study(direction='maximize', study_name=f"extreme_lgbm_{dataset_name}")
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best AUC: {study.best_value}")
        print(f"Best Params: {study.best_params}")
        
        best_params = study.best_params
        best_params['objective'] = 'binary'
        best_params['metric'] = 'auc'
        # Ensure boosting type is there if not suggested (it was hardcoded in obj)
        best_params['boosting_type'] = 'gbdt'
        
        # 6. Seed Averaging (Bagging)
        print("\n--- Starting Seed Averaging (10 Seeds) ---")
        
        seeds = range(42, 52) # 42 to 51
        test_probs_sum = np.zeros(len(X_test))
        val_probs_sum = np.zeros(len(X_train)) # OOF for threshold
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Main CV for threshold optimization
        
        # For Threshold Optimization, we need clean OOF predictions.
        # To do Bagging properly with OOF, we should bag INSIDE the CV? 
        # Or Bag over full train?
        # Standard Bagging: Train 10 full models on Full Train, Predict Test.
        # But we need Threshold. We need OOF preds for Threshold.
        # Strategy: 
        # For each Seed:
        #   Generate OOF preds (using 5-fold).
        #   Predict Test set.
        # Accumulate OOF preds and Test preds.
        # Final OOF = Avg(OOF_seed_i). Final Test = Avg(Test_seed_i).
        
        for seed in seeds:
            print(f"Training Seed {seed}...")
            
            # OOF Generation for this seed
            seed_oof = np.zeros(len(X_train))
            
            # We vary the seed of the KFold too? Or just the Model?
            # Varying model seed is enough for LGBM.
            # Varying KFold seed gives even more robust OOF.
            # Let's fix KFold seed (42) so folds are stable, but vary Model seed.
            
            seed_model_params = best_params.copy()
            seed_model_params['random_state'] = seed
            
            # Train on Full Train for Test Preds
            full_model = lgb.LGBMClassifier(n_jobs=-1, **seed_model_params)
            full_model.fit(X_train, y_train, categorical_feature=cat_features)
            seed_test_pred = full_model.predict_proba(X_test)[:, 1]
            test_probs_sum += seed_test_pred
            
            # OOF for Threshold
            for tr_idx, val_idx in skf.split(X_train, y_train):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_va = X_train.iloc[val_idx]
                
                fold_model = lgb.LGBMClassifier(n_jobs=-1, **seed_model_params)
                fold_model.fit(X_tr, y_tr, categorical_feature=cat_features)
                seed_oof[val_idx] = fold_model.predict_proba(X_va)[:, 1]
            
            val_probs_sum += seed_oof
            
        # Average
        final_test_probs = test_probs_sum / len(seeds)
        final_oof_probs = val_probs_sum / len(seeds)
        
        # Optimize Threshold
        best_th, best_f1_oof = optimize_threshold(y_train, final_oof_probs)
        auc_oof = roc_auc_score(y_train, final_oof_probs)
        
        print(f"\n--- Final Bagged Ensemble Results ---")
        print(f"OOF AUC: {auc_oof:.4f}")
        print(f"Best Threshold (OOF): {best_th:.2f}")
        print(f"OOF F1: {best_f1_oof:.4f}")
        
        # Test Eval
        final_preds = (final_test_probs >= best_th).astype(int)
        test_auc = roc_auc_score(y_test, final_test_probs)
        test_acc = accuracy_score(y_test, final_preds)
        
        print(f"\n=== TEST SET PERFORMANCE ===")
        print(f"AUC: {test_auc:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, final_preds))

if __name__ == "__main__":
    main()
