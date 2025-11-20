"""
Dataset Statistics Controller

Controller for dataset statistics endpoints.
"""
from backend.services.dataset_statistics_service import dataset_statistics_service
from backend.schemas.stats import (
    DatasetOverviewResponse,
    DemographicsResponse,
    ClinicalStatsResponse,
    CorrelationResponse,
    HighRiskProfilesResponse
)


class DatasetStatisticsController:
    """Controller for dataset statistics operations"""
    
    @staticmethod
    def get_overview() -> DatasetOverviewResponse:
        """
        Get dataset overview statistics
        
        Returns:
            DatasetOverviewResponse with overview statistics
        """
        data = dataset_statistics_service.get_overview()
        return DatasetOverviewResponse(**data)
    
    @staticmethod
    def get_demographics() -> DemographicsResponse:
        """
        Get demographic statistics
        
        Returns:
            DemographicsResponse with demographic statistics
        """
        data = dataset_statistics_service.get_demographics()
        return DemographicsResponse(**data)
    
    @staticmethod
    def get_clinical_stats() -> ClinicalStatsResponse:
        """
        Get clinical statistics
        
        Returns:
            ClinicalStatsResponse with clinical statistics
        """
        data = dataset_statistics_service.get_clinical_stats()
        return ClinicalStatsResponse(**data)
    
    @staticmethod
    def get_correlations() -> CorrelationResponse:
        """
        Get correlation matrix and top risk factors
        
        Returns:
            CorrelationResponse with correlation data
        """
        data = dataset_statistics_service.get_correlations()
        return CorrelationResponse(**data)
    
    @staticmethod
    def get_high_risk_profiles() -> HighRiskProfilesResponse:
        """
        Get high-risk profiles
        
        Returns:
            HighRiskProfilesResponse with high-risk profiles
        """
        data = dataset_statistics_service.get_high_risk_profiles()
        return HighRiskProfilesResponse(**data)


# Global instance
dataset_statistics_controller = DatasetStatisticsController()

