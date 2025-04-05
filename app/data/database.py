# resume_matcher/data/database.py
import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from ..utils.logger import logger

Base = declarative_base()


class Database:
    """Database connection manager"""

    def __init__(self):
        self._engine = None
        self._session_factory = None

    def init_db(self, connection_string: Optional[str] = None):
        """Initialize database connection"""
        connection_string = connection_string or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/resume_matcher",
        )

        try:
            self._engine = create_engine(
                connection_string,
                pool_size=20,
                max_overflow=0,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            self._session_factory = scoped_session(
                sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations"""
        if not self._session_factory:
            self.init_db()

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database operation failed: {str(e)}")
            raise
        finally:
            session.close()

    def create_tables(self):
        """Create all database tables"""
        if not self._engine:
            self.init_db()

        try:
            Base.metadata.create_all(bind=self._engine)
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Table creation failed: {str(e)}")
            raise


# Global database instance
db = Database()
