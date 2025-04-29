from __future__ import annotations
from enum import Enum

from pydantic import BaseModel, ConfigDict

class TokenType(Enum):
    bearer = 'bearer'

class AuthResponse(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    access_token: str
    refresh_token: str
    token_type: TokenType
    expires_in: int
