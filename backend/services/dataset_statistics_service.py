"""
Dataset Statistics Service

Service for calculating statistics from the original stroke dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from backend.config import settings
import logging

logger = logging.getLogger(__name__)


class DatasetStatisticsService:
    """Service for dataset statistics calculations"""
    
    def __init__(self):
        self._dataset: Optional[pd.DataFrame] = None
        self._dataset_paths = [
            Path(__file__).parent.parent.parent / "src" / "data" / "stroke_dataset.csv",
            Path(__file__).parent.parent.parent / "data" / "stroke_dataset.csv",
            Path("/app") / "src" / "data" / "stroke_dataset.csv",  # Docker
            Path("/app") / "data" / "stroke_dataset.csv",  # Docker
        ]
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load the original dataset CSV"""
        if self._dataset is not None:
            return self._dataset
        
        for path in self._dataset_paths:
            if path.exists():
                logger.info(f"Loading dataset from: {path}")
                self._dataset = pd.read_csv(path)
                return self._dataset
        
        raise FileNotFoundError(f"Dataset not found. Tried: {[str(p) for p in self._dataset_paths]}")
    
    def get_overview(self) -> Dict[str, Any]:
        """Get dataset overview statistics"""
        df = self._load_dataset()
        
        stroke_counts = df['stroke'].value_counts()
        total_samples = len(df)
        stroke_cases = stroke_counts.get(1, 0)
        no_stroke_cases = stroke_counts.get(0, 0)
        
        return {
            "total_samples": int(total_samples),
            "total_features": int(df.shape[1] - 1),  # Exclude target
            "stroke_cases": int(stroke_cases),
            "no_stroke_cases": int(no_stroke_cases),
            "class_balance": {
                "stroke": round(stroke_cases / total_samples * 100, 1),
                "no_stroke": round(no_stroke_cases / total_samples * 100, 1)
            },
            "missing_values": int(df.isnull().sum().sum())
        }
    
    def get_demographics(self) -> Dict[str, Any]:
        """Get demographic statistics"""
        df = self._load_dataset()
        
        # Age distribution
        age_ranges = [
            {"min": 0, "max": 20, "label": "0-20"},
            {"min": 21, "max": 40, "label": "21-40"},
            {"min": 41, "max": 60, "label": "41-60"},
            {"min": 61, "max": 80, "label": "61-80"},
            {"min": 81, "max": 200, "label": "81+"}
        ]
        
        age_distribution = []
        for age_range in age_ranges:
            mask = (df['age'] >= age_range['min']) & (df['age'] <= age_range['max'])
            subset = df[mask]
            count = len(subset)
            stroke_count = subset['stroke'].sum() if count > 0 else 0
            stroke_rate = (stroke_count / count * 100) if count > 0 else 0.0
            
            age_distribution.append({
                "range": age_range['label'],
                "count": int(count),
                "stroke_rate": round(stroke_rate, 1)
            })
        
        # Gender statistics
        gender_stats = {}
        for gender in df['gender'].unique():
            subset = df[df['gender'] == gender]
            count = len(subset)
            stroke_count = subset['stroke'].sum()
            stroke_rate = (stroke_count / count * 100) if count > 0 else 0.0
            
            gender_stats[gender] = {
                "count": int(count),
                "stroke_rate": round(stroke_rate, 1)
            }
        
        # Marital status (ever_married)
        marital_stats = {}
        for status in df['ever_married'].unique():
            subset = df[df['ever_married'] == status]
            count = len(subset)
            stroke_count = subset['stroke'].sum()
            stroke_rate = (stroke_count / count * 100) if count > 0 else 0.0
            
            marital_stats[status] = {
                "count": int(count),
                "stroke_rate": round(stroke_rate, 1)
            }
        
        return {
            "age": {
                "mean": round(float(df['age'].mean()), 1),
                "median": round(float(df['age'].median()), 1),
                "std": round(float(df['age'].std()), 1),
                "distribution": age_distribution
            },
            "gender": gender_stats,
            "marital_status": marital_stats
        }
    
    def get_clinical_stats(self) -> Dict[str, Any]:
        """Get clinical statistics"""
        df = self._load_dataset()
        
        # Hypertension
        hypertension_stats = {}
        for value in [0, 1]:
            subset = df[df['hypertension'] == value]
            count = len(subset)
            stroke_count = subset['stroke'].sum()
            stroke_rate = (stroke_count / count * 100) if count > 0 else 0.0
            
            key = "present" if value == 1 else "absent"
            hypertension_stats[key] = {
                "count": int(count),
                "stroke_rate": round(stroke_rate, 1)
            }
        
        # Heart disease
        heart_disease_stats = {}
        for value in [0, 1]:
            subset = df[df['heart_disease'] == value]
            count = len(subset)
            stroke_count = subset['stroke'].sum()
            stroke_rate = (stroke_count / count * 100) if count > 0 else 0.0
            
            key = "present" if value == 1 else "absent"
            heart_disease_stats[key] = {
                "count": int(count),
                "stroke_rate": round(stroke_rate, 1)
            }
        
        # Glucose levels
        glucose_ranges = [
            {"min": 0, "max": 100, "label": "<100"},
            {"min": 100, "max": 125, "label": "100-125"},
            {"min": 126, "max": 200, "label": "126-200"},
            {"min": 200, "max": 1000, "label": ">200"}
        ]
        
        glucose_distribution = []
        for glucose_range in glucose_ranges:
            mask = (df['avg_glucose_level'] >= glucose_range['min']) & (df['avg_glucose_level'] < glucose_range['max'])
            subset = df[mask]
            count = len(subset)
            stroke_count = subset['stroke'].sum() if count > 0 else 0
            stroke_rate = (stroke_count / count * 100) if count > 0 else 0.0
            
            glucose_distribution.append({
                "range": glucose_range['label'],
                "count": int(count),
                "stroke_rate": round(stroke_rate, 1)
            })
        
        # BMI categories
        bmi_categories = [
            {"min": 0, "max": 18.5, "name": "Underweight (<18.5)"},
            {"min": 18.5, "max": 25, "name": "Normal (18.5-24.9)"},
            {"min": 25, "max": 30, "name": "Overweight (25-29.9)"},
            {"min": 30, "max": 100, "name": "Obese (30+)"}
        ]
        
        bmi_distribution = []
        for bmi_cat in bmi_categories:
            mask = (df['bmi'] >= bmi_cat['min']) & (df['bmi'] < bmi_cat['max'])
            subset = df[mask]
            count = len(subset)
            stroke_count = subset['stroke'].sum() if count > 0 else 0
            stroke_rate = (stroke_count / count * 100) if count > 0 else 0.0
            
            bmi_distribution.append({
                "name": bmi_cat['name'],
                "count": int(count),
                "stroke_rate": round(stroke_rate, 1)
            })
        
        # Smoking status
        smoking_stats = {}
        for status in df['smoking_status'].unique():
            subset = df[df['smoking_status'] == status]
            count = len(subset)
            stroke_count = subset['stroke'].sum()
            stroke_rate = (stroke_count / count * 100) if count > 0 else 0.0
            
            smoking_stats[status] = {
                "count": int(count),
                "stroke_rate": round(stroke_rate, 1)
            }
        
        return {
            "hypertension": hypertension_stats,
            "heart_disease": heart_disease_stats,
            "avg_glucose_level": {
                "mean": round(float(df['avg_glucose_level'].mean()), 1),
                "median": round(float(df['avg_glucose_level'].median()), 1),
                "distribution": glucose_distribution
            },
            "bmi": {
                "mean": round(float(df['bmi'].mean()), 1),
                "categories": bmi_distribution
            },
            "smoking_status": smoking_stats
        }
    
    def get_correlations(self) -> Dict[str, Any]:
        """Get correlation matrix and top risk factors"""
        df = self._load_dataset()
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        
        corr_matrix = df[numeric_cols].corr()
        
        # Get correlations with stroke
        stroke_correlations = {}
        if 'stroke' in corr_matrix.index:
            for col in corr_matrix.columns:
                if col != 'stroke':
                    corr_value = corr_matrix.loc['stroke', col]
                    stroke_correlations[f"{col}_stroke"] = round(float(corr_value), 3)
        
        # Get other relevant correlations
        other_correlations = {}
        if 'age' in corr_matrix.index:
            if 'hypertension' in corr_matrix.columns:
                other_correlations["age_hypertension"] = round(float(corr_matrix.loc['age', 'hypertension']), 3)
            if 'avg_glucose_level' in corr_matrix.columns:
                other_correlations["age_glucose"] = round(float(corr_matrix.loc['age', 'avg_glucose_level']), 3)
        
        if 'bmi' in corr_matrix.index and 'avg_glucose_level' in corr_matrix.columns:
            other_correlations["bmi_glucose"] = round(float(corr_matrix.loc['bmi', 'avg_glucose_level']), 3)
        
        # Combine correlations
        all_correlations = {**stroke_correlations, **other_correlations}
        
        # Get top risk factors (highest correlation with stroke)
        top_risk_factors = []
        for key, value in sorted(stroke_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            feature = key.replace('_stroke', '')
            importance = "High" if abs(value) > 0.15 else "Medium" if abs(value) > 0.1 else "Low"
            top_risk_factors.append({
                "feature": feature,
                "correlation": value,
                "importance": importance
            })
        
        return {
            "correlation_matrix": all_correlations,
            "top_risk_factors": top_risk_factors[:10]  # Top 10
        }
    
    def get_high_risk_profiles(self) -> Dict[str, Any]:
        """Get high-risk profiles identified in the dataset"""
        df = self._load_dataset()
        
        profiles = []
        
        # Profile 1: Elderly with Hypertension
        profile1_mask = (df['age'] > 65) & (df['hypertension'] == 1)
        profile1_count = profile1_mask.sum()
        if profile1_count > 0:
            profile1_subset = df[profile1_mask]
            profile1_stroke_rate = (profile1_subset['stroke'].sum() / profile1_count * 100)
            profiles.append({
                "id": 1,
                "name": "Elderly with Hypertension",
                "criteria": "age > 65 AND hypertension = 1",
                "count": int(profile1_count),
                "stroke_rate": round(profile1_stroke_rate, 1),
                "avg_risk_score": round(profile1_stroke_rate / 100, 2)
            })
        
        # Profile 2: High Glucose + Heart Disease
        profile2_mask = (df['avg_glucose_level'] > 200) & (df['heart_disease'] == 1)
        profile2_count = profile2_mask.sum()
        if profile2_count > 0:
            profile2_subset = df[profile2_mask]
            profile2_stroke_rate = (profile2_subset['stroke'].sum() / profile2_count * 100)
            profiles.append({
                "id": 2,
                "name": "High Glucose + Heart Disease",
                "criteria": "avg_glucose_level > 200 AND heart_disease = 1",
                "count": int(profile2_count),
                "stroke_rate": round(profile2_stroke_rate, 1),
                "avg_risk_score": round(profile2_stroke_rate / 100, 2)
            })
        
        # Profile 3: Obese Elderly Smokers
        profile3_mask = (df['age'] > 60) & (df['bmi'] > 30) & (df['smoking_status'] == 'smokes')
        profile3_count = profile3_mask.sum()
        if profile3_count > 0:
            profile3_subset = df[profile3_mask]
            profile3_stroke_rate = (profile3_subset['stroke'].sum() / profile3_count * 100)
            profiles.append({
                "id": 3,
                "name": "Obese Elderly Smokers",
                "criteria": "age > 60 AND bmi > 30 AND smoking_status = 'smokes'",
                "count": int(profile3_count),
                "stroke_rate": round(profile3_stroke_rate, 1),
                "avg_risk_score": round(profile3_stroke_rate / 100, 2)
            })
        
        return {"profiles": profiles}


# Global instance
dataset_statistics_service = DatasetStatisticsService()

