# app/schemas/register_request.py
from pydantic import BaseModel, EmailStr, constr, model_validator, Field
from typing    import Optional, Annotated
from datetime  import date

from app.schemas.shared import (
    Role, Sex, PatientIn, DoctorIn
)


class RegisterRequest(BaseModel):
    email:    EmailStr
    password: Annotated[str, Field(min_length=8, max_length=128)]

    # exactly one of these ↓ must be supplied ────────────────────────────
    patient_profile: Optional[PatientIn] = None
    doctor_profile : Optional[DoctorIn]  = None
    # optional manual override; if omitted we’ll infer it from the profile
    role: Optional[Role] = None

    # ─────────────────────────────────────────────────────────────────────
    # Cross-field validation (Pydantic v2 style)
    # ─────────────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _exactly_one_profile(cls, model):
        if (model.patient_profile is None) == (model.doctor_profile is None):
            # either both None  OR  both not None  → invalid
            raise ValueError(
                "Provide **either** patient_profile **or** doctor_profile (not both)"
            )
        return model
