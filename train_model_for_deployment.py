# Model Training Script for Streamlit Deployment
# Run this script to generate the model files needed for the Streamlit app

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_model():
    """Train the wine quality model and save all necessary files"""
    
    print("Loading dataset...")


    # Load the dataset

    
    df = pd.read_csv('winequality-red-selected-missing.csv')
    
    print(f"Dataset loaded: {df.shape}")
    
    # Handle missing values if any
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Create binary target variable
    df['is_good_quality'] = (df['quality'] >= 7).astype(int)
    
    print("Target distribution:")
    print(df['is_good_quality'].value_counts())
    
    # Prepare features and target
    X = df.drop(['quality', 'is_good_quality'], axis=1)
    y = df['is_good_quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining model...")
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = []
    for i, (feature, importance) in enumerate(zip(X.columns, model.feature_importances_)):
        feature_importance.append({
            'feature': feature,
            'importance': importance
        })
    
    # Sort by importance
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
    
    print("\nTop 5 Feature Importances:")
    for i, item in enumerate(feature_importance[:5]):
        print(f"{i+1}. {item['feature']}: {item['importance']:.4f}")
    
    # Save model
    joblib.dump(model, 'wine_quality_model.joblib')
    print("\n✅ Model saved as 'wine_quality_model.joblib'")
    
    # Save metadata
    model_metadata = {
        'features': list(X.columns),
        'n_features': len(X.columns),
        'model_type': 'RandomForestClassifier',
        'accuracy': accuracy,
        'class_names': ['Not Good Quality', 'Good Quality'],
        'feature_importance': feature_importance,
        'target_distribution': df['is_good_quality'].value_counts().to_dict()
    }
    
    joblib.dump(model_metadata, 'model_metadata.joblib')
    print("✅ Model metadata saved as 'model_metadata.joblib'")
    
    print(f"\nFiles created for Streamlit deployment:")
    print("- wine_quality_model.joblib")
    print("- model_metadata.joblib")
    print("- Make sure you also have 'winequality-red.csv' in your deployment folder")
    
    return model, model_metadata

if __name__ == "__main__":
    model, metadata = train_and_save_model()