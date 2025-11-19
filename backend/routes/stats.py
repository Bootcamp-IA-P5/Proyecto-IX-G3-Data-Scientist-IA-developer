"""
Statistics endpoints

Routes for statistics and analytics.
"""
from fastapi import APIRouter
from backend.controllers.stats_controller import stats_controller
from backend.controllers.dataset_statistics_controller import dataset_statistics_controller
from backend.schemas import (
    StatsOverviewResponse,
    RiskDistributionResponse,
    ModelComparisonResponse,
    DashboardResponse
)
from backend.schemas.stats import (
    DatasetOverviewResponse,
    DemographicsResponse,
    ClinicalStatsResponse,
    CorrelationResponse,
    HighRiskProfilesResponse
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


# Dataset Statistics Endpoints
@router.get("/statistics/overview", response_model=DatasetOverviewResponse)
async def get_dataset_overview() -> DatasetOverviewResponse:
    """
    Get dataset overview statistics
    
    Returns:
        DatasetOverviewResponse with dataset overview
    """
    return dataset_statistics_controller.get_overview()


@router.get("/statistics/demographics", response_model=DemographicsResponse)
async def get_demographics() -> DemographicsResponse:
    """
    Get demographic statistics from the dataset
    
    Returns:
        DemographicsResponse with demographic statistics
    """
    return dataset_statistics_controller.get_demographics()


@router.get("/statistics/clinical", response_model=ClinicalStatsResponse)
async def get_clinical_stats() -> ClinicalStatsResponse:
    """
    Get clinical statistics from the dataset
    
    Returns:
        ClinicalStatsResponse with clinical statistics
    """
    return dataset_statistics_controller.get_clinical_stats()


@router.get("/statistics/correlations", response_model=CorrelationResponse)
async def get_correlations() -> CorrelationResponse:
    """
    Get correlation matrix and top risk factors
    
    Returns:
        CorrelationResponse with correlation data
    """
    return dataset_statistics_controller.get_correlations()


@router.get("/statistics/high-risk-profiles", response_model=HighRiskProfilesResponse)
async def get_high_risk_profiles() -> HighRiskProfilesResponse:
    """
    Get high-risk profiles identified in the dataset
    
    Returns:
        HighRiskProfilesResponse with high-risk profiles
    """
    return dataset_statistics_controller.get_high_risk_profiles()

