# app/db/models/user.py
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import relationship
from app.db.base import Base
from .doctor_salary import DoctorSalaryModel


class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'patient', 'doctor', 'admin'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # one-to-one links
    patient_profile = relationship(
        "PatientModel",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    doctor_profile = relationship(
        "DoctorModel",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    financial_profile = relationship(
        "DoctorSalaryModel",  # Use the class name as a string to avoid import issues here
        back_populates="user",
        uselist=False,  # One-to-one
        cascade="all, delete-orphan",  # If user is deleted, their salary info is also deleted
    )
