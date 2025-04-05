# apscheduler/jobstores/sqlalchemy.py
from typing import Dict, List, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ..job import Job
from .base import BaseJobStore, JobLookupError
import pickle
import logging

Base = declarative_base()
logger = logging.getLogger("apscheduler.jobstores.sqlalchemy")


class JobTable(Base):
    __tablename__ = "apscheduler_jobs"

    id = Column(String(191), primary_key=True)
    next_run_time = Column(DateTime, index=True)
    job_state = Column(Text, nullable=False)


class SQLAlchemyJobStore(BaseJobStore):
    """Job store that stores jobs in a SQL database using SQLAlchemy."""

    def __init__(
        self, url=None, engine=None, tablename="apscheduler_jobs", **engine_options
    ):
        super().__init__()
        if url and engine:
            raise ValueError("Either 'url' or 'engine' can be specified, not both")

        self.engine = engine or create_engine(url, **engine_options)
        self.tablename = tablename
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

    def start(self, scheduler, alias):
        super().start(scheduler, alias)
        logger.info("Starting SQLAlchemy job store [%s]", alias)

    def add_job(self, job: Job):
        session = self.Session()
        try:
            job_state = pickle.dumps(
                job.__getstate__(), protocol=pickle.HIGHEST_PROTOCOL
            )
            job_row = JobTable(
                id=job.id, next_run_time=job.next_run_time, job_state=job_state
            )
            session.add(job_row)
            session.commit()
        except Exception as e:
            session.rollback()
            raise ValueError(f"Failed to add job {job.id}: {str(e)}")
        finally:
            session.close()

    def update_job(self, job: Job):
        session = self.Session()
        try:
            job_state = pickle.dumps(
                job.__getstate__(), protocol=pickle.HIGHEST_PROTOCOL
            )
            job_row = session.query(JobTable).filter_by(id=job.id).first()
            if not job_row:
                raise JobLookupError(job.id)

            job_row.next_run_time = job.next_run_time
            job_row.job_state = job_state
            session.commit()
        except Exception as e:
            session.rollback()
            raise ValueError(f"Failed to update job {job.id}: {str(e)}")
        finally:
            session.close()

    def remove_job(self, job_id: str):
        session = self.Session()
        try:
            job_row = session.query(JobTable).filter_by(id=job_id).first()
            if not job_row:
                raise JobLookupError(job_id)

            session.delete(job_row)
            session.commit()
        except Exception as e:
            session.rollback()
            raise ValueError(f"Failed to remove job {job_id}: {str(e)}")
        finally:
            session.close()

    def load_jobs(self) -> List[Job]:
        session = self.Session()
        try:
            jobs = []
            for job_row in session.query(JobTable).all():
                try:
                    job_state = pickle.loads(job_row.job_state)
                    job = Job.__new__(Job)
                    job.__setstate__(job_state)
                    job._scheduler = self.scheduler
                    job._jobstore_alias = self.alias
                    jobs.append(job)
                except Exception as e:
                    logger.warning("Failed to load job %s: %s", job_row.id, str(e))
            return jobs
        finally:
            session.close()

    def get_job(self, job_id: str) -> Optional[Job]:
        session = self.Session()
        try:
            job_row = session.query(JobTable).filter_by(id=job_id).first()
            if not job_row:
                return None

            job_state = pickle.loads(job_row.job_state)
            job = Job.__new__(Job)
            job.__setstate__(job_state)
            job._scheduler = self.scheduler
            job._jobstore_alias = self.alias
            return job
        except Exception as e:
            logger.error("Failed to get job %s: %s", job_id, str(e))
            return None
        finally:
            session.close()

    def remove_all_jobs(self):
        session = self.Session()
        try:
            session.query(JobTable).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            raise ValueError(f"Failed to remove all jobs: {str(e)}")
        finally:
            session.close()

    def __repr__(self):
        return f"<SQLAlchemyJobStore (url={self.engine.url})>"
