import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import sys

# Logging removed to allow console output
# sys.stdout = log_file
# sys.stderr = log_file

print("Starting 01_build_datasets.py")

def medical_feature_engineering(df_input):
    """
    Applies domain-specific feature engineering.
    """
    print("Applying FE...")
    df_eng = df_input.copy()
    
    # Age conversion
    df_eng['Age_Year'] = df_eng['Age'] / 365.25
    
    # Unit conversions
    df_eng['bili_umolL'] = df_eng['Bilirubin'] * 17.1
    df_eng['alb_gL'] = df_eng['Albumin'] * 10
    df_eng['plt_1000uL'] = df_eng['Platelets'] / 1000
    
    # ALBI Score
    df_eng['ALBI'] = (np.log10(df_eng['bili_umolL']) * 0.66) + (df_eng['alb_gL'] * -0.085)
    
    # PALBI Score  
    df_eng['PALBI'] = (df_eng['ALBI'] * 1.0) + (df_eng['plt_1000uL'] * -0.04)
    
    # APRI
    df_eng['APRI'] = (df_eng['SGOT'] / 40) / df_eng['plt_1000uL']
    
    print("FE done.")
    return df_eng

def main():
    try:
        # 1. Load Data
        data_path = '../data/liver_cirrhosis_deduped.csv'
        print(f"Loading data from {data_path}")
        if not os.path.exists(data_path):
            print(f"Error: Data not found at {data_path}")
            return

        df = pd.read_csv(data_path)
        
        # Drop ID
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
            
        df = df.dropna(subset=['Stage'])
        
        # Filter Stages 1, 2, 3
        df_b = df[df['Stage'].isin([1, 2, 3])].copy()
        
        # Target Transformation: 1,2 -> 0, 3 -> 1
        df_b['Stage'] = df_b['Stage'].apply(lambda x: 0 if x in [1, 2] else 1)
        
        print(f"Total processed samples: {len(df_b)}")
        print(f"Class distribution: \n{df_b['Stage'].value_counts()}")

        # 2. Train / Test Split (Stratified 80/20)
        # IMPORTANT: Test set is LOCKED here.
        X = df_b.drop('Stage', axis=1)
        y = df_b['Stage']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Save combined Train/Test for easier handling in scripts (Target included)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Save Test set immediately
        test_file = 'data/processed/test.parquet'
        print(f"Saving test set to {test_file}")
        test_df.to_parquet(test_file)
        print("Test set saved.")

        # 3. Log Transformation (On Train Only - or rather transformation logic applied)
        # Note: For tree models log transform isn't strictly necessary but we follow the plan.
        log_transform_cols = ['Bilirubin', 'Copper', 'Alk_Phos', 'Tryglicerides', 'SGOT', 'Prothrombin', 'Cholesterol']
        
        print(f"Applying log transform to {log_transform_cols}")
        train_df_log = train_df.copy()
        for col in log_transform_cols:
            if col in train_df_log.columns:
                train_df_log[col] = np.log1p(train_df_log[col])
                
        # 4. Feature Engineering
        train_df_fe = medical_feature_engineering(train_df_log)
        
        # 5. Filter Columns as Requested
        # Global Removals: Metadata + Intermediate FE calculation columns
        global_drop_cols = [
            'N_Days', 'Status', 'Drug', 'Age', 'Sex',           # Metadata/Irrelevant
            'Age_Year', 'bili_umolL', 'alb_gL', 'plt_1000uL'    # Intermediate FE columns
        ]
        
        train_df_fe = train_df_fe.drop(columns=[c for c in global_drop_cols if c in train_df_fe.columns], errors='ignore')
        
        # Identify Column Sets
        all_cols = [c for c in train_df_fe.columns if c != 'Stage']
        
        # Define Groups
        score_cols = ['ALBI', 'PALBI', 'APRI']
        raw_score_inputs = ['Bilirubin', 'Albumin', 'Platelets', 'SGOT']
        
        # Dataset Definitions:
        # A: All features (Base + Scores)
        cols_A = all_cols
        
        # B: Base only (No Scores)
        cols_B = [c for c in all_cols if c not in score_cols]
        
        # C: Scores + (Base - Raw Inputs) -> "Remove variables used to make ALBI, PALBI"
        cols_C = [c for c in all_cols if c not in raw_score_inputs]
        
        # Create Dataframes (include Target in all)
        df_A = train_df_fe[cols_A + ['Stage']].copy()
        df_B = train_df_fe[cols_B + ['Stage']].copy()
        df_C = train_df_fe[cols_C + ['Stage']].copy()
        
        # Save Datasets
        print("Saving A, B, C datasets...")
        print(f"Dataset A columns ({len(cols_A)}): {cols_A}")
        print(f"Dataset B columns ({len(cols_B)}): {cols_B}")
        print(f"Dataset C columns ({len(cols_C)}): {cols_C}")
        
        df_A.to_parquet('data/processed/A.parquet')
        df_B.to_parquet('data/processed/B.parquet')
        df_C.to_parquet('data/processed/C.parquet')
        
        print("Datasets A, B, C saved to data/processed/")
        
        # Save Metadata
        def get_num_cols(df, exclude=['Stage']):
            return df.select_dtypes(include=[np.number]).columns.difference(exclude).tolist()

        meta = {
            'num_cols': {
                'A': get_num_cols(df_A),
                'B': get_num_cols(df_B),
                'C': get_num_cols(df_C)
            },
            'features': {
                'A': cols_A,
                'B': cols_B,
                'C': cols_C
            },
            'log_transform_cols': log_transform_cols
        }
        
        with open('data/processed/dataset_meta.json', 'w') as f:
            json.dump(meta, f, indent=4)
            
        print("Metadata saved to data/processed/dataset_meta.json")
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()

if __name__ == "__main__":
    main()
