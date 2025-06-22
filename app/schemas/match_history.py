# app/schemas/match_history.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# --- Base Schema: Defines the core, shared fields ---
class MatchHistoryBase(BaseModel):
    """
    Base schema containing common fields for match history records.
    Includes the link to the parsed resume.
    """

    user_id: int
    parsed_resume_id: Optional[int] = None  # The ID linking to the saved parsed resume
    resume_filename: Optional[str] = None  # Filename might not always be present
    job_ids: List[int] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Assuming results are a list of dicts


# --- Create Schema: Used for validating data when CREATING a new record ---
class MatchHistoryCreate(MatchHistoryBase):
    """
    Schema used specifically for validating the input data required
    to create a new match history record. Inherits all fields from Base.
    """

    # No additional fields are needed beyond those in MatchHistoryBase for creation input
    pass


# --- Update Schema: Used for validating data when UPDATING an existing record ---
# (Optional, define if you have update endpoints)
class MatchHistoryUpdate(BaseModel):
    """
    Schema used for partial updates. All fields are optional.
    """

    user_id: Optional[int] = None
    parsed_resume_id: Optional[int] = None
    resume_filename: Optional[str] = None
    job_ids: Optional[List[int]] = None
    preferences: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None


# --- Read Schema: Used for representing data when READING records from the DB ---
# This is typically used as the response_model for GET requests or nested in other responses.
class MatchHistoryRead(MatchHistoryBase):
    """
    Schema representing a complete match history record as read from the database.
    Includes database-generated fields like 'id' and 'created_at'.
    """

    id: int  # The unique database ID
    created_at: datetime  # The timestamp when the record was created

    class Config:
        from_attributes = True  # Enable creating this schema from an ORM object (SQLAlchemy model instance)


class MatchHistory(BaseModel):
    """
    (Optional) Specific schema if needed, potentially inheriting from Base
    or defining its own fields. Ensure consistency with MatchHistoryBase.
    """

    id: int  # ID is required here, assuming it's for reading existing records
    user_id: int
    parsed_resume_id: Optional[int] = None  # Added here too for consistency
    resume_filename: Optional[str] = None
    job_ids: List[int] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)  # Changed to List[Dict]
    created_at: datetime

    class Config:
        from_attributes = True
        # orm_mode = True # Pydantic v1

    # The custom model_validate is generally not needed with Config.from_attributes = True
    # Pydantic handles the ORM object mapping directly.
    # You can remove this unless you have specific validation logic beyond field mapping.
    # @classmethod
    # def model_validate(cls, obj):
    #     # ... (custom validation if necessary) ...
    #     # If just mapping fields, Config.from_attributes handles it.
    #     # If you keep this, ensure parsed_resume_id is mapped:
    #     # parsed_resume_id=getattr(obj, 'parsed_resume_id', None),
    #     # ... other fields ...
    #     return super().model_validate(obj) # Use super() for standard validation
