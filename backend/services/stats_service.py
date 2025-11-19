"""
Statistics service for tracking predictions and model performance.

This service stores prediction history in memory (for development).
In production, this should be replaced with a database.
"""
from typing import List, Dict, Optional
from collections import defaultdict
from backend.schemas.prediction import PredictionResponse


class StatsService:
    """
    Service for managing prediction statistics
    """
    
    def __init__(self):
        """Initialize the stats service"""
        self.prediction_history: List[PredictionResponse] = []
    
    def add_prediction(self, prediction: PredictionResponse):
        """
        Add a prediction to the history
        
        Args:
            prediction: PredictionResponse to add
        """
        self.prediction_history.append(prediction)
    
    def add_batch_predictions(self, predictions: List[PredictionResponse]):
        """
        Add multiple predictions to the history
        
        Args:
            predictions: List of PredictionResponse to add
        """
        self.prediction_history.extend(predictions)
    
    def get_overview_stats(self) -> Dict:
        """
        Get overview statistics
        
        Returns:
            Dictionary with overview statistics
        """
        if not self.prediction_history:
            return {
                "total_predictions": 0,
                "stroke_predictions": 0,
                "no_stroke_predictions": 0,
                "average_probability": 0.0
            }
        
        total = len(self.prediction_history)
        stroke_count = sum(1 for p in self.prediction_history if p.prediction == 1)
        no_stroke_count = total - stroke_count
        avg_prob = sum(p.probability for p in self.prediction_history) / total
        
        return {
            "total_predictions": total,
            "stroke_predictions": stroke_count,
            "no_stroke_predictions": no_stroke_count,
            "average_probability": round(avg_prob, 4)
        }
    
    def get_risk_distribution(self) -> Dict:
        """
        Get risk distribution statistics
        
        Returns:
            Dictionary with risk distribution
        """
        if not self.prediction_history:
            return {
                "low_risk": 0,
                "medium_risk": 0,
                "high_risk": 0,
                "distribution": {}
            }
        
        low = sum(1 for p in self.prediction_history if p.confidence == "Low")
        medium = sum(1 for p in self.prediction_history if p.confidence == "Medium")
        high = sum(1 for p in self.prediction_history if p.confidence == "High")
        
        distribution = {
            "Low": low,
            "Medium": medium,
            "High": high
        }
        
        return {
            "low_risk": low,
            "medium_risk": medium,
            "high_risk": high,
            "distribution": distribution
        }
    
    def clear_history(self):
        """Clear prediction history"""
        self.prediction_history.clear()


# Global instance
stats_service = StatsService()

