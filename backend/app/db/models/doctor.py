# app/db/models/doctor.py
from sqlalchemy import Column, Integer, String, ForeignKey, Date
from sqlalchemy.orm import relationship
from app.db.base import Base

class DoctorModel(Base):
    __tablename__ = "doctors"

    user_id    = Column(Integer,
                        ForeignKey("users.id", ondelete="CASCADE"),
                        primary_key=True)

    first_name = Column(String(50), nullable=False)
    last_name  = Column(String(50), nullable=False)
    specialty  = Column(String(100),nullable=False)
    dob        = Column(Date, nullable=False)
    sex        = Column(String(1), nullable=False)
    phone      = Column(String(20), nullable=False)

    user = relationship("UserModel", back_populates="doctor_profile")
