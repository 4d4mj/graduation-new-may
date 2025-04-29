from __future__ import annotations
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    email: EmailStr
    password: Annotated[str, Field(min_length=8, max_length=128)]
