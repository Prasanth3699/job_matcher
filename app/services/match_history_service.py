from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pytz

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError  # Catch specific DB errors

# --- Import the specific model ---
from ..models.match_history import MatchHistory as MatchHistoryModel

# --- Import the CORRECTED and REFACTORED schemas ---
from ..schemas.match_history import MatchHistoryRead, MatchHistoryCreate

# ---------------------------------------------------

# Assuming logger is configured correctly in your project
from ..utils.logger import logger

# Define timezone if used (e.g., for the model default lambda)
IST = pytz.timezone("Asia/Kolkata")


class MatchHistoryService:
    @staticmethod
    def create_match_history(
        db: Session,
        user_id: int,
        resume_filename: Optional[str],  # Allow None if schema does
        job_ids: List[int],
        preferences: Dict[str, Any],
        results: List[Dict[str, Any]],
        parsed_resume_id: Optional[int] = None,
    ) -> MatchHistoryRead:  # Return type is the Read schema
        """
        Creates and saves a new match history record to the database after validation.

        Args:
            db: The SQLAlchemy database session.
            user_id: The ID of the user performing the match.
            resume_filename: The name of the resume file used (optional).
            job_ids: A list of job IDs matched against.
            preferences: The user preferences used for matching.
            results: The list of match results (as dictionaries).
            parsed_resume_id: The optional ID of the associated record in the parsed_resumes table.

        Returns:
            The created MatchHistory record, validated against the MatchHistoryRead schema.

        Raises:
            HTTPException: If input validation fails (422) or database saving fails (500).
        """
        # 1. Validate input data using the Create schema
        try:
            history_data = MatchHistoryCreate(
                user_id=user_id,
                resume_filename=resume_filename,
                job_ids=job_ids,
                preferences=preferences,
                results=results,
                parsed_resume_id=parsed_resume_id,
            )
        except Exception as validation_error:
            logger.error(
                f"Pydantic validation failed for MatchHistoryCreate data: {validation_error}"
            )
            # Use 422 for validation errors
            raise HTTPException(
                status_code=422,
                detail=f"Invalid match history input data: {validation_error}",
            )

        # 2. Create SQLAlchemy model instance using validated data
        try:
            logger.info(
                f"Creating match history entry for user_id: {user_id}, parsed_resume_id: {parsed_resume_id}"
            )

            # Use model_dump() to get dict from Pydantic obj for SQLAlchemy constructor
            db_match = MatchHistoryModel(**history_data.model_dump())

            # Note: created_at is handled by the model's `default` - no need to set it here.

            # 3. Add to session, commit, and refresh
            db.add(db_match)
            db.commit()
            db.refresh(db_match)  # Load DB-generated values like ID and created_at
            logger.info(f"Match history saved successfully with ID: {db_match.id}")

            # 4. Validate the result against the Read schema before returning
            # This ensures the data returned matches the expected output structure
            return MatchHistoryRead.model_validate(db_match)

        except SQLAlchemyError as e:
            db.rollback()  # Rollback the transaction on database errors
            logger.error(
                f"Database error saving match history for user {user_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="A database error occurred while saving match history.",
            )
        except Exception as e:
            # Catch any other unexpected errors during DB interaction or validation
            db.rollback()
            logger.error(
                f"Unexpected error saving match history for user {user_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while saving match history.",
            )

    @staticmethod
    def get_user_match_history(
        db: Session, user_id: int, limit: int = 10
    ) -> List[MatchHistoryRead]:  # Return type is a list of Read schemas
        """
        Retrieves a list of recent match history records for a specific user.

        Args:
            db: The SQLAlchemy database session.
            user_id: The ID of the user whose history to retrieve.
            limit: The maximum number of history records to return.

        Returns:
            A list of MatchHistory records, validated against the MatchHistoryRead schema.

        Raises:
            HTTPException: If database retrieval fails (500).
        """
        try:
            logger.info(
                f"Fetching match history for user_id: {user_id}, limit: {limit}"
            )
            # Query the database
            matches = (
                db.query(MatchHistoryModel)
                .filter(MatchHistoryModel.user_id == user_id)
                .order_by(MatchHistoryModel.created_at.desc())
                .limit(limit)
                .all()
            )
            logger.info(
                f"Found {len(matches)} match history records for user_id: {user_id}"
            )

            # Validate each retrieved record against the Read schema
            return [MatchHistoryRead.model_validate(match) for match in matches]

        except SQLAlchemyError as e:
            logger.error(
                f"Database error fetching match history for user {user_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="A database error occurred while retrieving match history.",
            )
        except Exception as e:
            logger.error(
                f"Unexpected error fetching match history for user {user_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while retrieving match history.",
            )
