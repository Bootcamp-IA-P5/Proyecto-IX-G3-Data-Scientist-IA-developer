"""
SQLAlchemy models for database tables.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database.connection import Base


class PatientData(Base):
    """
    Table that stores patient input data.
    Stores RAW values (untransformed) as they come from the frontend.
    """
    __tablename__ = "patient_data"

    # Primary column
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Automatic timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # --- Patient data (10 features) ---
    # These are the RAW values coming from the frontend
    age = Column(Integer, nullable=False, comment="Patient age")
    gender = Column(String(10), nullable=False, comment="Male/Female/Other")
    hypertension = Column(Boolean, nullable=False, comment="0=No, 1=Yes")
    heart_disease = Column(Boolean, nullable=False, comment="0=No, 1=Yes")
    ever_married = Column(String(3), nullable=False, comment="Yes/No")
    work_type = Column(String(20), nullable=False, comment="Private/Self-employed/Govt_job/children/Never_worked")
    residence_type = Column(String(10), nullable=False, comment="Urban/Rural")
    avg_glucose_level = Column(Float, nullable=False, comment="Average glucose level")
    bmi = Column(Float, nullable=False, comment="Body mass index")
    smoking_status = Column(String(20), nullable=False, comment="formerly smoked/never smoked/smokes/Unknown")
    
    # Relationship with predictions (one patient can have multiple predictions)
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<PatientData(id={self.id}, age={self.age}, gender={self.gender})>"


class Prediction(Base):
    """
    Table that stores prediction results.
    A prediction is always associated with a PatientData record.
    """
    __tablename__ = "predictions"

    # Primary column
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Foreign Key to patient_data
    patient_data_id = Column(
        Integer, 
        ForeignKey("patient_data.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Associated patient ID"
    )
    
    # Automatic timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # --- Prediction results ---
    model_name = Column(String(50), nullable=False, comment="Model name used (e.g.: logistic_regression)")
    prediction = Column(Integer, nullable=False, comment="Result: 0=No stroke, 1=Stroke")
    probability = Column(Float, nullable=False, comment="Stroke probability (0.0 - 1.0)")
    risk_level = Column(String(10), nullable=False, comment="Low/Medium/High")
    
    # Inverse relationship with patient_data
    patient = relationship("PatientData", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction(id={self.id}, patient_id={self.patient_data_id}, prediction={self.prediction}, risk={self.risk_level})>"
