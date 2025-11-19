"""
CRUD operations (Create, Read, Update, Delete) for the database.
"""
from sqlalchemy.orm import Session
from typing import Optional, List
from backend.database.models import PatientData, Prediction
from backend.schemas.prediction import PredictionRequest


def create_patient_data(db: Session, patient_request: PredictionRequest) -> PatientData:
    """
    Creates a new patient data record.
    
    Args:
        db: Database session
        patient_request: Patient data from the request
    
    Returns:
        PatientData: Created object with assigned ID
    """
    patient = PatientData(
        age=patient_request.age,
        gender=patient_request.gender,
        hypertension=patient_request.hypertension,
        heart_disease=patient_request.heart_disease,
        ever_married=patient_request.ever_married,
        work_type=patient_request.work_type,
        residence_type=patient_request.Residence_type,
        avg_glucose_level=patient_request.avg_glucose_level,
        bmi=patient_request.bmi,
        smoking_status=patient_request.smoking_status
    )
    
    db.add(patient)
    db.commit()
    db.refresh(patient)  # Gets the generated ID
    return patient


def create_prediction(
    db: Session,
    patient_data_id: int,
    model_name: str,
    prediction: int,
    probability: float,
    risk_level: str
) -> Prediction:
    """
    Creates a new prediction record.
    
    Args:
        db: Database session
        patient_data_id: Associated patient ID
        model_name: Model name used
        prediction: Prediction result (0 or 1)
        probability: Calculated probability
        risk_level: Risk level (Low/Medium/High)
    
    Returns:
        Prediction: Created object with assigned ID
    """
    prediction_obj = Prediction(
        patient_data_id=patient_data_id,
        model_name=model_name,
        prediction=prediction,
        probability=probability,
        risk_level=risk_level
    )
    
    db.add(prediction_obj)
    db.commit()
    db.refresh(prediction_obj)
    return prediction_obj


def get_patient_by_id(db: Session, patient_id: int) -> Optional[PatientData]:
    """
    Gets a patient by their ID.
    
    Args:
        db: Database session
        patient_id: Patient ID
    
    Returns:
        PatientData or None if it doesn't exist
    """
    return db.query(PatientData).filter(PatientData.id == patient_id).first()


def get_all_patients(db: Session, skip: int = 0, limit: int = 100) -> List[PatientData]:
    """
    Gets a paginated list of patients.
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        List of PatientData
    """
    return db.query(PatientData).offset(skip).limit(limit).all()


def get_predictions_by_patient(db: Session, patient_id: int) -> List[Prediction]:
    """
    Gets all predictions for a patient.
    
    Args:
        db: Database session
        patient_id: Patient ID
    
    Returns:
        List of Prediction
    """
    return db.query(Prediction).filter(Prediction.patient_data_id == patient_id).all()


def get_all_predictions(db: Session, skip: int = 0, limit: int = 100) -> List[Prediction]:
    """
    Gets a paginated list of all predictions.
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        List of Prediction
    """
    return db.query(Prediction).offset(skip).limit(limit).all()


def get_latest_predictions(db: Session, limit: int = 10) -> List[Prediction]:
    """
    Gets the most recent predictions.
    
    Args:
        db: Database session
        limit: Number of predictions to return
    
    Returns:
        List of Prediction ordered by descending date
    """
    return db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()


def delete_patient(db: Session, patient_id: int) -> bool:
    """
    Deletes a patient and all their predictions (cascade).
    
    Args:
        db: Database session
        patient_id: Patient ID to delete
    
    Returns:
        True if deleted, False if it didn't exist
    """
    patient = get_patient_by_id(db, patient_id)
    if patient:
        db.delete(patient)
        db.commit()
        return True
    return False
