"""
Preprocessing service for transforming API input data to model format.

This service applies the same preprocessing pipeline used during training:
- Feature engineering
- Encoding (LabelEncoder + One-Hot Encoding)
- Scaling (StandardScaler)
"""
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler


class PreprocessingService:
    """
    Service for preprocessing prediction input data
    """
    
    def __init__(self):
        """Initialize the preprocessing service"""
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.expected_columns: Optional[list] = None
        self.data_dir = self._get_data_dir()
        self._load_preprocessing_artifacts()
    
    def _get_data_dir(self) -> Path:
        """
        Get the path to the data directory
        
        Returns:
            Path to data directory
        """
        possible_paths = [
            Path(__file__).parent.parent.parent / "backend" / "data",
            Path(__file__).parent.parent.parent / "data",
            Path(__file__).parent / "data",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Default to backend/data
        return Path(__file__).parent.parent.parent / "backend" / "data"
    
    def _load_preprocessing_artifacts(self):
        """Load scaler and determine expected columns"""
        scaler_path = self.data_dir / "scaler.pkl"
        
        if not scaler_path.exists():
            print(f"⚠️ Warning: Scaler not found at {scaler_path}")
            return
        
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load a sample to get expected columns
            sample_path = self.data_dir / "X_train_balanced.pkl"
            if sample_path.exists():
                with open(sample_path, 'rb') as f:
                    sample_data = pickle.load(f)
                    if hasattr(sample_data, 'columns'):
                        self.expected_columns = list(sample_data.columns)
                    else:
                        # If it's a numpy array, we need to infer from scaler
                        self.expected_columns = None
        except Exception as e:
            print(f"⚠️ Warning: Error loading preprocessing artifacts: {e}")
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        df_processed = df.copy()
        
        # 1. Categorías de edad
        df_processed['age_group'] = pd.cut(
            df_processed['age'],
            bins=[0, 18, 40, 60, 100],
            labels=['Child', 'Young_Adult', 'Adult', 'Senior']
        )
        
        # 2. Categorías de glucosa
        df_processed['glucose_category'] = pd.cut(
            df_processed['avg_glucose_level'],
            bins=[0, 100, 125, 300],
            labels=['Normal', 'Prediabetes', 'Diabetes']
        )
        
        # 3. Categorías de BMI
        df_processed['bmi_category'] = pd.cut(
            df_processed['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # 4. Smoking binario
        df_processed['has_smoked'] = df_processed['smoking_status'].apply(
            lambda x: 1 if x in ['smokes', 'formerly smoked'] else 0
        )
        
        # 5. Risk score compuesto
        df_processed['risk_score'] = (
            df_processed['age'] * 0.3 +
            df_processed['hypertension'] * 20 +
            df_processed['heart_disease'] * 25 +
            df_processed['avg_glucose_level'] * 0.1 +
            df_processed['bmi'] * 0.5
        )
        
        # 6. Interacciones
        df_processed['age_x_hypertension'] = df_processed['age'] * df_processed['hypertension']
        df_processed['age_x_heart_disease'] = df_processed['age'] * df_processed['heart_disease']
        df_processed['glucose_x_bmi'] = df_processed['avg_glucose_level'] * df_processed['bmi']
        
        return df_processed
    
    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encoding transformations (LabelEncoder + One-Hot Encoding)
        
        Args:
            df: Input dataframe
            
        Returns:
            Encoded dataframe
        """
        df_encoded = df.copy()
        
        # Label Encoding para binarias
        binary_features = ['ever_married']
        for feat in binary_features:
            if feat in df_encoded.columns:
                le = LabelEncoder()
                # Fit with known values
                le.fit(['Yes', 'No'])
                df_encoded[feat] = le.transform(df_encoded[feat])
        
        # One-Hot Encoding para categóricas
        categorical_features = ['work_type', 'smoking_status', 'age_group', 'glucose_category', 'bmi_category']
        categorical_features = [f for f in categorical_features if f in df_encoded.columns]
        
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_features, drop_first=True)
        
        return df_encoded
    
    def preprocess(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for prediction
        
        Args:
            data: Dictionary with input features (from PredictionRequest)
            
        Returns:
            Preprocessed DataFrame ready for model prediction (with column names)
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Remove features that are not used (gender, Residence_type)
        df = df.drop(columns=['gender', 'Residence_type'], errors='ignore')
        
        # Apply feature engineering
        df = self._apply_feature_engineering(df)
        
        # Apply encoding
        df = self._apply_encoding(df)
        
        # Ensure all expected columns are present
        if self.expected_columns:
            # Add missing columns with 0 values
            for col in self.expected_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match expected order
            df = df[self.expected_columns]
        
        # Apply scaling
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Please ensure preprocessing data exists.")
        
        X_scaled = self.scaler.transform(df)
        
        # Convert back to DataFrame to maintain column names
        X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
        
        return X_scaled_df
    
    def preprocess_batch(self, data_list: list) -> pd.DataFrame:
        """
        Preprocess batch of input data
        
        Args:
            data_list: List of dictionaries with input features
            
        Returns:
            Preprocessed DataFrame ready for batch prediction (with column names)
        """
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Remove features that are not used
        df = df.drop(columns=['gender', 'Residence_type'], errors='ignore')
        
        # Apply feature engineering
        df = self._apply_feature_engineering(df)
        
        # Apply encoding
        df = self._apply_encoding(df)
        
        # Ensure all expected columns are present
        if self.expected_columns:
            # Add missing columns with 0 values
            for col in self.expected_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match expected order
            df = df[self.expected_columns]
        
        # Apply scaling
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Please ensure preprocessing data exists.")
        
        X_scaled = self.scaler.transform(df)
        
        # Convert back to DataFrame to maintain column names
        X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
        
        return X_scaled_df


# Global instance
preprocessing_service = PreprocessingService()

