"""
Error schemas
"""
from typing import Optional
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    detail: Optional[str] = None

