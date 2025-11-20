"""
Statistics controller

Handles statistics and analytics logic.
"""
from typing import Dict
from backend.services.stats_service import stats_service
from backend.services.model_service import model_service
from backend.controllers.model_controller import model_controller
from backend.controllers.health_controller import health_controller
from backend.schemas import (
    StatsOverviewResponse,
    RiskDistributionResponse,
    ModelComparisonResponse,
    DashboardResponse
)


class StatsController:
    """Controller for statistics operations"""
    
    @staticmethod
    def get_overview() -> StatsOverviewResponse:
        """
        Get overview statistics
        
        Returns:
            StatsOverviewResponse with overview statistics
        """
        stats = stats_service.get_overview_stats()
        return StatsOverviewResponse(**stats)
    
    @staticmethod
    def get_risk_distribution() -> RiskDistributionResponse:
        """
        Get risk distribution statistics
        
        Returns:
            RiskDistributionResponse with risk distribution
        """
        stats = stats_service.get_risk_distribution()
        return RiskDistributionResponse(**stats)
    
    @staticmethod
    def compare_models() -> ModelComparisonResponse:
        """
        Compare available models based on their metrics
        
        Returns:
            ModelComparisonResponse with model comparison
        """
        models = model_service.get_available_models()
        
        if not models:
            return ModelComparisonResponse(
                models=[],
                metrics={},
                best_model=None
            )
        
        metrics_dict = {}
        best_model = None
        best_recall = -1
        
        for model_name in models:
            try:
                model_info = model_controller.get_model_info(model_name)
                if model_info.metrics:
                    metrics_dict[model_name] = model_info.metrics
                    
                    # Determine best model by RECALL (most important in medical context)
                    # Recall = ability to detect all stroke cases (minimize false negatives)
                    recall = model_info.metrics.get("recall", 0.0)
                    if recall > best_recall:
                        best_recall = recall
                        best_model = model_name
            except Exception:
                # Skip models that can't be loaded
                continue
        
        return ModelComparisonResponse(
            models=models,
            metrics=metrics_dict,
            best_model=best_model
        )
    
    @staticmethod
    def get_dashboard() -> DashboardResponse:
        """
        Get comprehensive dashboard data combining all relevant information
        
        Returns:
            DashboardResponse with all dashboard information
        """
        # Get system status
        status = health_controller.get_status()
        
        # Get prediction statistics
        overview = StatsController.get_overview()
        
        # Get risk distribution
        risk_dist = StatsController.get_risk_distribution()
        
        # Get model comparison
        comparison = StatsController.compare_models()
        
        # Get best model details
        best_model_info = None
        best_model_metrics = None
        best_model_type = None
        
        if comparison.best_model:
            try:
                best_model_info = model_controller.get_model_info(comparison.best_model)
                best_model_metrics = best_model_info.metrics
                best_model_type = best_model_info.model_type
            except Exception:
                pass
        
        # Create performance summary
        performance_summary = {
            "total_models": len(comparison.models),
            "models_with_metrics": len(comparison.metrics),
            "average_accuracy": 0.0,
            "average_recall": 0.0,
            "average_auc_roc": 0.0
        }
        
        if comparison.metrics:
            accuracies = [m.get("accuracy", 0) for m in comparison.metrics.values() if m.get("accuracy")]
            recalls = [m.get("recall", 0) for m in comparison.metrics.values() if m.get("recall")]
            aucs = [m.get("auc_roc", 0) for m in comparison.metrics.values() if m.get("auc_roc")]
            
            if accuracies:
                performance_summary["average_accuracy"] = round(sum(accuracies) / len(accuracies), 4)
            if recalls:
                performance_summary["average_recall"] = round(sum(recalls) / len(recalls), 4)
            if aucs:
                performance_summary["average_auc_roc"] = round(sum(aucs) / len(aucs), 4)
        
        return DashboardResponse(
            # System status
            api_status=status.api_status,
            models_loaded=status.models_loaded,
            total_models=len(status.available_models),
            available_models=status.available_models,
            
            # Prediction statistics
            total_predictions=overview.total_predictions,
            stroke_predictions=overview.stroke_predictions,
            no_stroke_predictions=overview.no_stroke_predictions,
            average_probability=overview.average_probability,
            
            # Risk distribution
            risk_distribution=risk_dist.distribution,
            
            # Best model
            best_model=comparison.best_model,
            best_model_metrics=best_model_metrics,
            best_model_type=best_model_type,
            
            # Model comparison
            model_comparison=comparison.metrics,
            
            # Performance summary
            model_performance_summary=performance_summary
        )


# Global instance
stats_controller = StatsController()

