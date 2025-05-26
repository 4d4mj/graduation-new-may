# backend/app/db/models/doctor_salary.py
from sqlalchemy import Column, Integer, ForeignKey, Numeric, Date, String, Text
from sqlalchemy.orm import relationship
from app.db.base import Base
from sqlalchemy.dialects.postgresql import (
    MONEY,
)  # Using MONEY for currency, or Numeric for more precision


class DoctorSalaryModel(Base):
    __tablename__ = "doctor_salaries"

    # Link to the User ID of the doctor. This is the primary key for this table
    # and a foreign key to the users table.
    doctor_user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )

    base_salary_annual = Column(
        Numeric(12, 2), nullable=False
    )  # Example: 12 digits, 2 decimal places

    last_bonus_amount = Column(Numeric(10, 2), nullable=True)
    last_bonus_date = Column(Date, nullable=True)
    last_bonus_reason = Column(Text, nullable=True)

    last_raise_percentage = Column(Numeric(5, 2), nullable=True)  # Example: 3.75%
    last_raise_date = Column(Date, nullable=True)
    last_raise_reason = Column(Text, nullable=True)

    next_review_period = Column(String(50), nullable=True)  # e.g., "Q3 2025"

    # Define the relationship back to the UserModel (one-to-one with DoctorSalaryModel)
    # This 'user' attribute will allow you to access the DoctorSalaryModel from a UserModel instance
    # if you set up the reverse relationship in UserModel.
    user = relationship("UserModel", back_populates="financial_profile")

    def __repr__(self):
        return f"<DoctorSalaryModel(doctor_user_id={self.doctor_user_id}, salary={self.base_salary_annual})>"
