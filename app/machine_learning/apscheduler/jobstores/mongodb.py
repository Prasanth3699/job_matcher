# apscheduler/jobstores/mongodb.py
from typing import Dict, List, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from ..job import Job
from .base import BaseJobStore, JobLookupError
import pickle
import logging

logger = logging.getLogger("apscheduler.jobstores.mongodb")


class MongoDBJobStore(BaseJobStore):
    """Job store that stores jobs in a MongoDB database."""

    def __init__(
        self,
        database="apscheduler",
        collection="jobs",
        host="localhost",
        port=27017,
        client=None,
        **connect_args,
    ):
        super().__init__()
        if client:
            self.client = client
        else:
            self.client = MongoClient(host=host, port=port, **connect_args)

        self.database = self.client[database]
        self.collection: Collection = self.database[collection]
        self.collection.create_index("next_run_time")

    def start(self, scheduler, alias):
        super().start(scheduler, alias)
        logger.info("Starting MongoDB job store [%s]", alias)

    def add_job(self, job: Job):
        job_state = pickle.dumps(job.__getstate__(), protocol=pickle.HIGHEST_PROTOCOL)
        document = {
            "_id": job.id,
            "next_run_time": job.next_run_time,
            "job_state": job_state,
        }
        try:
            self.collection.insert_one(document)
        except Exception as e:
            raise ValueError(f"Failed to add job {job.id}: {str(e)}")

    def update_job(self, job: Job):
        job_state = pickle.dumps(job.__getstate__(), protocol=pickle.HIGHEST_PROTOCOL)
        result = self.collection.update_one(
            {"_id": job.id},
            {"$set": {"next_run_time": job.next_run_time, "job_state": job_state}},
        )
        if result.matched_count == 0:
            raise JobLookupError(job.id)

    def remove_job(self, job_id: str):
        result = self.collection.delete_one({"_id": job_id})
        if result.deleted_count == 0:
            raise JobLookupError(job_id)

    def load_jobs(self) -> List[Job]:
        jobs = []
        for document in self.collection.find():
            try:
                job_state = pickle.loads(document["job_state"])
                job = Job.__new__(Job)
                job.__setstate__(job_state)
                job._scheduler = self.scheduler
                job._jobstore_alias = self.alias
                jobs.append(job)
            except Exception as e:
                logger.warning("Failed to load job %s: %s", document["_id"], str(e))
        return jobs

    def get_job(self, job_id: str) -> Optional[Job]:
        document = self.collection.find_one({"_id": job_id})
        if not document:
            return None

        try:
            job_state = pickle.loads(document["job_state"])
            job = Job.__new__(Job)
            job.__setstate__(job_state)
            job._scheduler = self.scheduler
            job._jobstore_alias = self.alias
            return job
        except Exception as e:
            logger.error("Failed to get job %s: %s", job_id, str(e))
            return None

    def remove_all_jobs(self):
        self.collection.delete_many({})

    def __repr__(self):
        return f"<MongoDBJobStore (database={self.database.name}, collection={self.collection.name})>"
