import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

def limpiar_tiempo(valor):
    """Limpia el formato sucio de tiempo 1970..."""
    try:
        valor_str = str(valor)
        horas = float(valor_str.split('.')[-1])
        return horas
    except:
        return np.nan

def load_and_clean_data(filepath, target_col):
    df = pd.read_csv(filepath)
    
    
    if df[target_col].dtype == 'O': 
        df['target_clean'] = df[target_col].apply(limpiar_tiempo)
        df = df.dropna(subset=['target_clean'])
        y = df['target_clean']
    else:
        y = df[target_col]
        
   
    features_num = ['distance_km', 'package_weight_kg', 'delivery_rating', 'delivery_cost']
    features_cat = ['vehicle_type', 'weather_condition', 'delivery_mode', 'region']
    
    
    available_cols = [c for c in features_num + features_cat if c in df.columns]
    X = df[available_cols]
    
    return X, y, features_num, features_cat

def build_pipeline(features_num, features_cat):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_num),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
        ])
    return preprocessor

def save_preprocessor(preprocessor, path):
    joblib.dump(preprocessor, path)
    print(f"Preprocessor guardado en {path}")

def load_preprocessor(path):
    return joblib.load(path)