"""
Application dependencies providing dependency injection for domain services and infrastructure.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession  # for async SQLAlchemy
from typing import AsyncGenerator

from app.db.session import get_db  # assumes canonical get_db below

from app.domain.matching.services import MatchingDomainService
from app.domain.matching.repositories import SQLAlchemyMatchingRepository
from app.domain.jobs.services import JobDomainService
from app.domain.jobs.repositories import SQLAlchemyJobRepository
from app.domain.users.services import UserDomainService
from app.domain.users.repositories import SQLAlchemyUserRepository


# --- DATABASE SESSION (use directly as a dependency) ---

# This is the canonical async DB session dependency for FastAPI
# with SQLAlchemy 1.4+ (async engine):
# (your app.db.session.get_db should look like this)
# ------------------------------------------------------
# async def get_db() -> AsyncGenerator[AsyncSession, None]:
#     async with AsyncSessionLocal() as session:
#         yield session
# ------------------------------------------------------

# --- REPOSITORY DEPENDENCIES ---


async def get_matching_repository(
    db: AsyncSession = Depends(get_db),
) -> SQLAlchemyMatchingRepository:
    """Get matching repository (async)."""
    return SQLAlchemyMatchingRepository(db)


async def get_job_repository(
    db: AsyncSession = Depends(get_db),
) -> SQLAlchemyJobRepository:
    """Get job repository (async)."""
    return SQLAlchemyJobRepository(db)


async def get_user_repository(
    db: AsyncSession = Depends(get_db),
) -> SQLAlchemyUserRepository:
    """Get user repository (async)."""
    return SQLAlchemyUserRepository(db)


# --- DOMAIN SERVICE DEPENDENCIES ---


async def get_matching_service(
    repository: SQLAlchemyMatchingRepository = Depends(get_matching_repository),
) -> MatchingDomainService:
    """Get matching domain service with repository dependency."""
    return MatchingDomainService(repository)


async def get_job_service(
    repository: SQLAlchemyJobRepository = Depends(get_job_repository),
) -> JobDomainService:
    """Get job domain service with repository dependency."""
    return JobDomainService(repository)


async def get_user_service(
    repository: SQLAlchemyUserRepository = Depends(get_user_repository),
) -> UserDomainService:
    """Get user domain service with repository dependency."""
    return UserDomainService(repository)


# --- LEGACY SERVICES (optional, for migration) ---


def get_legacy_resume_parser():
    from app.core.document_processing import ResumeParser

    return ResumeParser()


def get_legacy_resume_extractor():
    from app.core.document_processing import ResumeExtractor

    return ResumeExtractor()


def get_legacy_job_processor():
    from app.core.job_processing.service import JobProcessingService

    return JobProcessingService()


def get_legacy_matching_service():
    from app.core.matching.service import MatchingService

    return MatchingService()
