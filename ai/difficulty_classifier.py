import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_classifier():
    print('Loading dataset...')
    
    # Load dataset
    df = pd.read_excel('dataset_annotations.xlsx')
    
    # Features and labels
    X = df[['unique_chord_count', 'chord_transition_freq', 
             'tempo', 'num_sections']]
    y = df['difficulty']
    
    print(f'Dataset size: {len(df)} songs')
    print(f'Classes: {y.unique()}')
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    # Save model and encoder
    joblib.dump(model, 'difficulty_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    print('Model trained and saved successfully')
    print(f'Classes: {le.classes_}')
    
    return model, le

def classify_difficulty(features):
    
    # Load model if it exists
    if not os.path.exists('difficulty_model.pkl'):
        print('No model found - training now...')
        train_classifier()
    
    model = joblib.load('difficulty_model.pkl')
    le = joblib.load('label_encoder.pkl')
    
    
    # Build feature vector
    feature_vector = pd.DataFrame([[
        features.get('unique_chord_count', 5),
        features.get('chord_transition_freq', 0.1),
        features.get('tempo', 100),
        features.get('num_sections', 5)
    ]], columns=['unique_chord_count', 'chord_transition_freq', 
                 'tempo', 'num_sections'])
    
    # Predict
    prediction = model.predict(feature_vector)
    difficulty = le.inverse_transform(prediction)[0]
    
    # Get probability scores
    probabilities = model.predict_proba(feature_vector)[0]
    confidence = round(float(max(probabilities)) * 100, 1)
    
    return {
        'difficulty': difficulty,
        'confidence': confidence
    }