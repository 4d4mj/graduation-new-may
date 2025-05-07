# app/db/models/allergies.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class AllergyModel(Base):
    __tablename__ = "allergies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.user_id", ondelete="CASCADE"), nullable=False)
    substance = Column(String(100), nullable=False)
    reaction = Column(Text)
    severity = Column(String(20))  # Could be 'Mild', 'Moderate', 'Severe', etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship with patient
    patient = relationship("PatientModel", back_populates="allergies")
