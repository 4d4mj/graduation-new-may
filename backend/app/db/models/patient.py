# app/db/models/patient.py
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base

class PatientModel(Base):
    __tablename__ = "patients"

    # ← this column *is* required
    user_id    = Column(Integer,
                        ForeignKey("users.id", ondelete="CASCADE"),
                        primary_key=True)

    first_name = Column(String(50), nullable=False)
    last_name  = Column(String(50), nullable=False)
    dob        = Column(Date,        nullable=False)
    sex        = Column(String(1))
    phone      = Column(String(30))
    address    = Column(String(255))

    user = relationship("UserModel", back_populates="patient_profile")
    allergies = relationship("AllergyModel", back_populates="patient", cascade="all, delete-orphan")
