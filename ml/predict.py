import numpy as np
import pandas as pd
import joblib
import os

class BreastCancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load the trained model and scaler"""
        try:
            if os.path.exists('ml/breast_cancer_model.pkl'):
                self.model = joblib.load('ml/breast_cancer_model.pkl')
                print("✅ Model loaded successfully")
            else:
                print("❌ Model file not found")
                
            if os.path.exists('ml/scaler.pkl'):
                self.scaler = joblib.load('ml/scaler.pkl')
                print("✅ Scaler loaded successfully")
            else:
                print("❌ Scaler file not found")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def get_full_features(self, data):
        """
        Create a full feature vector with 55 features matching the training data
        """
        # Initialize all features to 0
        features = np.zeros(55)
        
        # Map of feature indices (you need to adjust these based on your actual training data)
        # These indices are examples - you must match your actual feature names
        
        # Numerical features (indices 0-3)
        features[0] = float(data.get('age', 50))           # Age
        features[1] = float(data.get('bmi', 25))           # BMI
        features[2] = float(data.get('tumor_size', 20))    # Tumor Size
        features[3] = float(data.get('inv_nodes', 0))      # Involved Nodes
        
        # Menopause (one-hot encoding)
        menopause = data.get('menopause', 'premeno')
        if menopause == 'premeno':
            features[4] = 1  # premeno
        elif menopause == 'perimeno':
            features[5] = 1  # perimeno
        elif menopause == 'postmeno':
            features[6] = 1  # postmeno
        
        # Metastasis
        metastasis = data.get('metastasis', 'no')
        features[7] = 1 if metastasis == 'yes' else 0
        
        # History
        history = data.get('history', 'no')
        features[8] = 1 if history == 'yes' else 0
        
        # Breast Side
        breast_side = data.get('breast_side', 'left')
        features[9] = 1 if breast_side == 'right' else 0
        
        # Breast Quadrant (one-hot encoding)
        quadrant = data.get('breast_quadrant', 'left_up')
        quadrant_map = {
            'left_up': 10,
            'left_low': 11,
            'right_up': 12,
            'right_low': 13,
            'central': 14
        }
        if quadrant in quadrant_map:
            features[quadrant_map[quadrant]] = 1
        
        # Note: Add all other feature mappings here based on your training data
        # For now, the remaining features (15-54) remain 0
        
        return features.reshape(1, -1)
    
    def predict(self, form_data):
        """Make prediction using the loaded model"""
        if self.model is None or self.scaler is None:
            return None, None
        
        try:
            # Get full feature vector (55 features)
            features = self.get_full_features(form_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            probability = self.model.predict_proba(features_scaled)[0][1]
            prediction = 1 if probability >= 0.5 else 0
            
            return prediction, probability
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None

# Create global instance
predictor = BreastCancerPredictor()