"""
Statistics endpoints

Routes for statistics and analytics.
"""
from fastapi import APIRouter
from backend.controllers.stats_controller import stats_controller
from backend.schemas import (
    StatsOverviewResponse,
    RiskDistributionResponse,
    ModelComparisonResponse,
    DashboardResponse
)

router = APIRouter()


@router.get("/stats/overview", response_model=StatsOverviewResponse)
async def get_stats_overview() -> StatsOverviewResponse:
    """
    Get overview statistics of predictions
    
    Returns:
        StatsOverviewResponse with overview statistics
    """
    return stats_controller.get_overview()


@router.get("/stats/risk-distribution", response_model=RiskDistributionResponse)
async def get_risk_distribution() -> RiskDistributionResponse:
    """
    Get risk distribution statistics
    
    Returns:
        RiskDistributionResponse with risk distribution
    """
    return stats_controller.get_risk_distribution()


@router.get("/stats/models/compare", response_model=ModelComparisonResponse)
async def compare_models() -> ModelComparisonResponse:
    """
    Compare available models based on their performance metrics
    
    Returns:
        ModelComparisonResponse with model comparison
    """
    return stats_controller.compare_models()


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard() -> DashboardResponse:
    """
    Get comprehensive dashboard data - combines system status, predictions, 
    risk distribution, and model information in a single response
    
    Returns:
        DashboardResponse with all dashboard information
    """
    return stats_controller.get_dashboard()

