import yaml
import sys
import os
import joblib
sys.path.append(os.getcwd())

from src.features.build_features import load_and_clean_data, build_pipeline, save_preprocessor
from src.models.model_arch import create_model

def train():
   
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("--- Cargando Datos ---")
    X, y, feats_num, feats_cat = load_and_clean_data(config['data']['raw_path'], config['data']['target_col'])
    
   
    print("--- Ajustando Preprocessor ---")
    preprocessor = build_pipeline(feats_num, feats_cat)
    X_processed = preprocessor.fit_transform(X)
    
 
    save_preprocessor(preprocessor, config['model']['preprocessor_path'])
    
   
    print("--- Entrenando Red Neuronal ---")
    input_shape = X_processed.shape[1]
    model = create_model(input_shape, learning_rate=config['training']['learning_rate'])
    
    model.fit(
        X_processed, y,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_split=0.2,
        verbose=1
    )
    
  
    model.save(config['model']['save_path'])
    print(f"Modelo guardado en {config['model']['save_path']}")

if __name__ == "__main__":
    train()