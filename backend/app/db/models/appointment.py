# app/db/models/appointment.py
from sqlalchemy import (
    Column,
    Integer,
    DateTime,
    String,
    Text,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.db.base import Base
from sqlalchemy.sql import func


class AppointmentModel(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    starts_at = Column(DateTime(timezone=True), nullable=False)
    ends_at = Column(DateTime(timezone=True), nullable=False)
    location = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    status = Column(
        String, default="scheduled", nullable=False
    )  # e.g., scheduled, completed, cancelled_by_patient, cancelled_by_doctor
    google_calendar_event_id = Column(
        String, nullable=True, index=True
    )  # Store GCal event ID

    # Define the unique constraint to avoid overlaps
    __table_args__ = (UniqueConstraint("doctor_id", "starts_at", name="unq_doc_time"),)

    # Relationships
    patient = relationship(
        "UserModel", foreign_keys=[patient_id], backref="patient_appointments"
    )
    doctor = relationship(
        "UserModel", foreign_keys=[doctor_id], backref="doctor_appointments"
    )
