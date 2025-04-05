from datetime import datetime
from typing import List, Dict, Any
from fastapi import HTTPException
from sqlalchemy.orm import Session
from ..models.match_history import MatchHistory as MatchHistoryModel

from ..models.match_history import MatchHistoryBase

# from app.schemas.match_history import MatchHistory


class MatchHistoryService:
    @staticmethod
    def create_match_history(
        db: Session,
        user_id: int,
        resume_filename: str,
        job_ids: List[int],
        preferences: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> MatchHistoryBase:
        try:
            db_match = MatchHistoryModel(
                user_id=user_id,
                resume_filename=resume_filename,
                job_ids=job_ids,
                preferences=preferences,
                results=results,
                created_at=datetime.utcnow(),
            )
            db.add(db_match)
            db.commit()
            db.refresh(db_match)
            # return MatchHistory.model_validate(db_match)
            return MatchHistoryBase.model_validate(db_match)
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500, detail=f"Failed to save match history: {str(e)}"
            )

    @staticmethod
    def get_user_match_history(
        db: Session, user_id: int, limit: int = 10
    ) -> List[MatchHistoryBase]:
        matches = (
            db.query(MatchHistoryModel)
            .filter(MatchHistoryModel.user_id == user_id)
            .order_by(MatchHistoryModel.created_at.desc())
            .limit(limit)
            .all()
        )
        return [MatchHistoryBase.model_validate(match) for match in matches]
