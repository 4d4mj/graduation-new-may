# app/db/models/doctor.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base

class DoctorModel(Base):
    __tablename__ = "doctors"

    user_id    = Column(Integer,
                        ForeignKey("users.id", ondelete="CASCADE"),
                        primary_key=True)

    specialty  = Column(String(100), nullable=False)

    user = relationship("UserModel", back_populates="doctor_profile")
