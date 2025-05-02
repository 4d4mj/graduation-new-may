# app/schemas/shared.py
from datetime import datetime, date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, EmailStr

class Sex(str, Enum):
    m = "M"
    f = "F"

class Role(str, Enum):
    doctor = "doctor"
    patient = "patient"
    admin = "admin"

class PatientOut(BaseModel):
    first_name: str
    last_name: str
    dob: date
    sex: Sex
    phone: str
    address: str

class DoctorOut(BaseModel):
    first_name: str
    last_name: str
    specialty: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    role: Role
    patient_profile: Optional[PatientOut] = None
    doctor_profile: Optional[DoctorOut] = None
    created_at: datetime


class PatientIn(PatientOut):        # inherits first_name … address …
    pass                            # <- nothing extra for now

class DoctorIn(DoctorOut):
    pass
