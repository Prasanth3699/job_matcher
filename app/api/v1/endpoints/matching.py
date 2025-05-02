from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from typing import List, Dict, Any, Optional  # Added Optional
import json
import uuid
from sqlalchemy.orm import Session
from pathlib import Path

from app.core.auth import get_current_user
from app.db.session import get_db
from app.services.job_service import JobService  # Assuming this exists and works
from app.services.match_history_service import MatchHistoryService

# Use the schema for response model if defined, otherwise base
from app.schemas.match_history import (
    MatchHistory,
    MatchHistoryBase,
)  # Ensure MatchHistoryBase/MatchHistory includes parsed_resume_id
from app.utils.logger import logger
from app.services.dependencies import (
    get_resume_parser,
    get_resume_extractor,
    get_job_processor,
    get_matching_service,
    # get_secure_resume_file, # This dependency wasn't used, commented out
)


from app.schemas.matching import (
    MatchErrorResponse,
    JobMatchResult,
    MatchResponse,
    JobDetails,
)
from app.utils.file_helpers import cleanup_temp_file

# Ensure the client is correctly imported
from app.services.profile_client import ProfileServiceClient

router = APIRouter(
    tags=["matching"],
    responses={404: {"description": "Not found"}},
)


# Helper function to safely convert values to string
def safe_str(val):
    """Safely converts a value to a string, returning empty string for None."""
    if val is None:
        return ""
    # Convert known non-str types to str
    if not isinstance(val, str):
        try:
            return str(val)
        except Exception:  # Catch potential errors during string conversion
            logger.warning(f"Could not convert value of type {type(val)} to string.")
            return ""
    return val


# Helper function to sanitize sequences (lists) into lists of non-empty strings
def sanitize_sequence(seq):
    """Ensure all elements in seq are non-empty strings, return cleaned list."""
    if not isinstance(seq, (list, tuple, set)):  # Check if it's an iterable sequence
        if seq is not None and safe_str(seq).strip():
            return [safe_str(seq).strip()]  # Handle single non-list items
        else:
            return []
    # Process actual sequences
    return [safe_str(x).strip() for x in seq if x is not None and safe_str(x).strip()]


@router.get(
    "/history", response_model=List[MatchHistoryBase]
)  # Use Base or specific response schema
async def get_match_history(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 10,
):
    """
    Retrieves the match history for the currently authenticated user.
    """
    user_id = current_user.get("user_id")
    if user_id is None:
        logger.error("User ID not found in token payload during history fetch.")
        raise HTTPException(
            status_code=401, detail="Invalid authentication token."
        )  # Or 500 if it's an internal issue

    return MatchHistoryService.get_user_match_history(db, user_id, limit)


@router.post(
    "/new-matchs",  # Consider renaming to /new-matches (plural consistency)
    response_model=MatchResponse,
    status_code=200,  # 200 OK is more standard for returning results of an operation
    # responses={
    #     200: {"model": MatchResponse, "description": "Matching successful"},
    #     400: {"model": MatchErrorResponse, "description": "Invalid input data."},
    #     401: {"description": "Authentication failed."},
    #     404: {
    #         "model": MatchErrorResponse,
    #         "description": "No matches found or jobs not found.",
    #     },
    #     500: {
    #         "model": MatchErrorResponse,
    #         "description": "Internal server error during matching.",
    #     },
    # },
)
async def new_match(
    resume_file: UploadFile = File(
        ..., description="The user's resume file (.pdf, .docx, etc.)."
    ),
    job_ids: str = Form(
        ..., description="Comma-separated string of job IDs to match against."
    ),
    preferences: str = Form(
        ...,
        description="JSON string containing user's job preferences (e.g., location, salary).",
    ),
    resume_parser=Depends(get_resume_parser),
    resume_extractor=Depends(get_resume_extractor),
    job_processor=Depends(get_job_processor),
    matching_service=Depends(get_matching_service),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Matches an uploaded resume against specified job postings based on user preferences.

    - Parses and extracts data from the resume.
    - **Saves the parsed resume details asynchronously via ProfileServiceClient.**
    - Fetches job details.
    - Processes resume and job data for matching.
    - Performs the matching calculation.
    - Formats and returns the match results.
    - **Saves the match attempt details (including the parsed resume ID if available) to history.**
    """
    temp_path: Optional[Path] = None
    parsed_resume_id: Optional[int] = (
        None  # Variable to store the ID from ProfileService
    )

    try:
        # --- 1. Input Validation ---
        logger.info(
            f"New match request received for user_id: {current_user.get('user_id')}"
        )
        if (
            not resume_file.filename
            or resume_file.size == 0
            or resume_file.size is None
        ):
            logger.warning("Invalid resume file uploaded.")
            raise HTTPException(
                status_code=400, detail="Invalid or empty resume file provided."
            )

        try:
            pref_data = json.loads(preferences)
            job_ids_list = [
                int(id_str.strip())
                for id_str in job_ids.split(",")
                if id_str.strip().isdigit()  # Ensure only digits
            ]
            if not job_ids_list:
                raise ValueError("No valid job IDs provided.")
            logger.debug(f"Parsed preferences: {pref_data}, Job IDs: {job_ids_list}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Invalid input format for preferences or job_ids: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format for preferences or job_ids: {e}",
            )

        user_id = current_user.get("user_id")
        auth_token = current_user.get("token")
        if not user_id or not auth_token:
            logger.error("Missing user_id or token in current_user object.")
            raise HTTPException(
                status_code=401, detail="Authentication token invalid or incomplete."
            )

        # --- 2. Resume Processing ---
        file_bytes = await resume_file.read()
        await resume_file.seek(0)  # Reset file pointer in case it's needed again

        # Save to a temporary file for parsing
        temp_path = Path(f"temp_{uuid.uuid4().hex}_{resume_file.filename}")
        try:
            with open(temp_path, "wb") as buffer:
                buffer.write(file_bytes)
            logger.info(f"Resume saved to temporary path: {temp_path}")

            # Parse and extract resume content
            raw_text, metadata = resume_parser.parse(temp_path)
            # Assuming resume_extractor returns an object with attributes
            resume_data_obj = resume_extractor.extract(raw_text, metadata)
            # Convert extracted data object to dict for processing and saving
            # Handle potential errors if resume_data_obj doesn't have __dict__
            try:
                parsed_resume_dict = vars(
                    resume_data_obj
                ).copy()  # Use copy to avoid modifying original object unexpectedly
            except TypeError:
                if isinstance(resume_data_obj, dict):
                    parsed_resume_dict = resume_data_obj.copy()
                else:
                    logger.error(
                        f"Resume extractor returned unexpected type: {type(resume_data_obj)}"
                    )
                    raise HTTPException(
                        status_code=500, detail="Internal error processing resume data."
                    )

            logger.info("Resume parsed and data extracted successfully.")

        except Exception as e:
            logger.error(f"Error during resume parsing/extraction: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to process resume file: {e}"
            )

        # --- 3. Push Parsed Resume to Profile Service (Asynchronously) ---
        # This call saves the resume file and its parsed data via a separate service.
        # We capture the response to get the database ID assigned by that service.
        parsed_resume_response = await ProfileServiceClient.push_parsed_resume(
            token=auth_token,
            filename=resume_file.filename,
            file_bytes=file_bytes,
            raw_text=raw_text,
            parsed_data=parsed_resume_dict,  # Send the dictionary form
            metadata=metadata,
        )

        # Extract the ID if the push was successful
        if parsed_resume_response and isinstance(parsed_resume_response.get("id"), int):
            parsed_resume_id = parsed_resume_response["id"]
            logger.info(
                f"Successfully pushed parsed resume via ProfileServiceClient. Received ID: {parsed_resume_id}"
            )
        else:
            # Log a warning if saving failed, but continue the matching process.
            # If saving the parsed resume is CRITICAL, you might raise an HTTPException here instead.
            logger.warning(
                "Failed to push parsed resume via ProfileServiceClient or did not receive a valid ID in response."
            )

        # --- 4. Enrich Resume Data with Preferences ---
        # Add user's preferences to the resume data for matching context.
        # Ensure this modification happens on the dictionary version.
        parsed_resume_dict["normalized_features"] = {
            "job_type_preference": pref_data.get("preferred_job_types", [""])[
                0
            ].lower(),
            "location_preference": pref_data.get("preferred_locations", [""])[
                0
            ].lower(),
            "salary_expectation": pref_data.get("salary_expectation", "not specified"),
            "target_title": pref_data.get("target_title", "").lower(),
            "preferred_companies": [
                c.lower() for c in pref_data.get("preferred_companies", [])
            ],
        }
        logger.debug("Added preferences to normalized_features in resume data.")

        # --- 5. Fetch and Process Job Postings ---
        logger.info(f"Fetching job details for IDs: {job_ids_list}")
        jobs = await JobService.fetch_jobs(job_ids_list, auth_token)
        if not jobs:
            logger.warning(f"No job details found for IDs: {job_ids_list}")
            raise HTTPException(
                status_code=404,
                detail="Could not find job details for the provided IDs.",
            )

        # Create a lookup for original job IDs based on apply_link (if apply_link is unique)
        # This helps map results back to the original IDs requested by the user.
        orig_id_by_apply_link = {}
        for j in jobs:
            job_data = j.model_dump() if hasattr(j, "model_dump") else dict(j)
            apply_link = safe_str(job_data.get("apply_link", ""))
            # Use job_id first, fallback to id
            original_job_id = job_data.get("job_id", job_data.get("id"))
            if apply_link and original_job_id is not None:
                orig_id_by_apply_link[apply_link] = original_job_id

        logger.debug(
            f"Original ID lookup created based on apply_link: {len(orig_id_by_apply_link)} entries"
        )

        processed_jobs = []
        logger.info(f"Processing {len(jobs)} fetched jobs for matching.")
        for job in jobs:
            job_dict = job.model_dump() if hasattr(job, "model_dump") else dict(job)

            # Map fields and sanitize data to match expected structure for matching service
            processed_job = {
                "job_title": safe_str(job_dict.get("job_title", "")),
                "company_name": safe_str(job_dict.get("company_name", "")),
                "location": safe_str(job_dict.get("location", "")),
                "job_type": safe_str(job_dict.get("job_type", "")),
                "apply_link": safe_str(job_dict.get("apply_link", "")),
                "skills": sanitize_sequence(job_dict.get("skills", [])),
                "requirements": sanitize_sequence(job_dict.get("requirements", [])),
                "responsibilities": sanitize_sequence(
                    job_dict.get("responsibilities", [])
                ),
                "benefits": sanitize_sequence(job_dict.get("benefits", [])),
                "qualifications": sanitize_sequence(job_dict.get("qualifications", [])),
                "description": safe_str(job_dict.get("description", "")),
                # Add other fields needed by the matching service directly if necessary
            }

            # Add normalized_features required by the matching/processing logic
            processed_job["normalized_features"] = {
                "job_type": safe_str(processed_job["job_type"]),
                "location": safe_str(processed_job["location"]),
                "salary": safe_str(
                    job_dict.get("salary", "")
                ),  # Get salary from original dict
                "company_name": safe_str(processed_job["company_name"]),
                # Ensure skills in normalized features are also sanitized
                "skills": sanitize_sequence(processed_job.get("skills", [])),
                "title": safe_str(processed_job["job_title"]),
            }
            processed_jobs.append(processed_job)

        # Further ensure list fields are present and correctly formatted (redundancy for safety)
        list_fields = [
            "skills",
            "requirements",
            "responsibilities",
            "benefits",
            "qualifications",
        ]
        for job in processed_jobs:
            for k in list_fields:
                job[k] = sanitize_sequence(
                    job.get(k)
                )  # Use sanitize_sequence for consistency
            if "normalized_features" in job and "skills" in job["normalized_features"]:
                job["normalized_features"]["skills"] = sanitize_sequence(
                    job["normalized_features"]["skills"]
                )

        # Batch process jobs if needed (e.g., vectorization, further normalization)
        if job_processor:
            processed_jobs = job_processor.process_job_batch(processed_jobs)
            logger.info("Job batch processed.")
        else:
            logger.warning("Job processor dependency not available.")

        # --- 6. Perform Matching ---
        logger.info("Performing matching between resume and processed jobs.")
        # Pass the dictionary form of the resume data and the list of processed jobs
        results = matching_service.match_resume_to_jobs(
            parsed_resume_dict,
            processed_jobs,
            top_n=len(processed_jobs),  # Match against all processed jobs initially
        )

        if not results:
            logger.info("No matches found by the matching service.")
            # Consider returning 200 OK with empty list instead of 404 if no technical error occurred
            # raise HTTPException(status_code=404, detail="No matches found for this resume and job selection.")
            return []  # Return empty list if no matches found

        logger.info(f"Matching service returned {len(results)} potential matches.")

        # --- 7. Format Results ---
        formatted_results: List[JobMatchResult] = []
        for result in results:
            # Safely access nested dictionaries and values
            job_details_from_match = result.get("job_details", {})
            apply_link = safe_str(job_details_from_match.get("apply_link", ""))
            # Look up the original ID using the apply link
            original_id = orig_id_by_apply_link.get(apply_link) if apply_link else None

            # Prepare data for JobDetails sub-schema
            job_details_data = {
                "job_title": safe_str(job_details_from_match.get("job_title", "")),
                "company_name": safe_str(
                    job_details_from_match.get("company_name", "")
                ),
                "location": safe_str(job_details_from_match.get("location", "")),
                "job_type": safe_str(job_details_from_match.get("job_type", "")),
                "apply_link": apply_link
                or None,  # Pass None if empty for HttpUrl validation
            }

            # Construct the JobMatchResult data
            formatted_result_data = {
                "job_id": result.get("job_id", ""),
                "original_job_id": original_id,  # <-- Set at top level
                "overall_score": result.get("overall_score", 0.0),
                "score_breakdown": result.get("score_breakdown", {}),
                "missing_skills": sanitize_sequence(result.get("missing_skills", [])),
                "matching_skills": sanitize_sequence(result.get("matching_skills", [])),
                "explanation": safe_str(result.get("explanation", "")),
                "job_details": job_details_data,  # Pass the prepared sub-dict
            }
            # Validate against Pydantic model before appending (optional but good practice)
            try:
                formatted_results.append(JobMatchResult(**formatted_result_data))
            except Exception as pydantic_error:
                logger.error(
                    f"Error validating formatted match result: {pydantic_error}",
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=500,
                    detail="Error formatting match results.",
                )

        # --- 8. Save Match History ---
        try:
            # Pass the captured parsed_resume_id to the history service
            logger.info(
                f"Saving match history for user {user_id}. Parsed Resume ID: {parsed_resume_id}"
            )
            MatchHistoryService.create_match_history(
                db=db,
                user_id=user_id,
                resume_filename=safe_str(resume_file.filename),
                job_ids=job_ids_list,  # The original list of requested IDs
                preferences=pref_data,
                results=[
                    res.model_dump() for res in formatted_results
                ],  # Save results as dicts
                parsed_resume_id=parsed_resume_id,  # Pass the ID here
            )
            logger.info("Match history saved successfully.")
        except Exception as history_exc:
            # Log error but don't fail the request just because history saving failed
            logger.error(f"Failed to save match history: {history_exc}", exc_info=True)

        # --- 9. Return Formatted Results AND parsed_resume_id ---
        logger.info(
            f"Returning {len(formatted_results)} formatted match results. Parsed Resume ID: {parsed_resume_id}"
        )
        # Return the new response structure
        response_data = {
            "matches": formatted_results,
            "parsed_resume_id": parsed_resume_id,
        }
        logger.info(f"Response data --------------: {response_data}")
        return response_data

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        logger.warning(
            f"HTTP Exception caught in new_match: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(
            f"Unexpected error during matching process: {str(e)}", exc_info=True
        )
        # Return a generic 500 error response
        raise HTTPException(
            status_code=500,
            # Use the MatchErrorResponse model if it's simple enough
            detail=MatchErrorResponse(
                detail="An unexpected error occurred during the matching process.",
                error_type="InternalServerError",
                context=str(e),  # Provide generic context
            ).model_dump(),
        )
    finally:
        # --- 10. Cleanup Temporary File ---
        # Schedule the temp file deletion to run in the background
        if temp_path and temp_path.exists():
            logger.info(f"Scheduling cleanup for temporary file: {temp_path}")
            background_tasks.add_task(cleanup_temp_file, temp_path)
