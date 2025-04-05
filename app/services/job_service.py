import httpx
from typing import List
import re
from bs4 import BeautifulSoup
import logging

from ..core.config import get_settings
from ..schemas.resume import Job

settings = get_settings()
logger = logging.getLogger(__name__)


class JobService:
    """Service for interacting with the Jobs API with enhanced content cleaning"""

    @staticmethod
    async def fetch_jobs(job_ids: List[int]) -> List[Job]:
        """Fetch job details from the Jobs API and clean the descriptions"""
        jobs = []

        if not job_ids:
            logger.warning("No job IDs provided to fetch")
            return jobs

        async with httpx.AsyncClient(base_url=settings.JOBS_SERVICE_URL) as client:
            for job_id in job_ids:
                try:
                    response = await client.get(f"/api/v1/jobs/{job_id}", timeout=10.0)

                    if response.status_code == 200:
                        job_data = response.json()

                        # Clean HTML from description if present
                        if "description" in job_data:
                            job_data["description"] = JobService._clean_html_content(
                                job_data["description"]
                            )

                        # Clean HTML from other fields if necessary
                        if "requirements" in job_data:
                            job_data["requirements"] = JobService._clean_html_content(
                                job_data["requirements"]
                            )

                        if "responsibilities" in job_data:
                            job_data["responsibilities"] = (
                                JobService._clean_html_content(
                                    job_data["responsibilities"]
                                )
                            )

                        jobs.append(Job(**job_data))
                    else:
                        logger.error(
                            f"Error fetching job {job_id}: Status {response.status_code}, Response: {response.text[:100]}"
                        )
                except httpx.RequestError as e:
                    logger.error(f"Request error fetching job {job_id}: {str(e)}")
                except httpx.TimeoutException:
                    logger.error(f"Timeout fetching job {job_id}")
                except Exception as e:
                    logger.error(f"Exception fetching job {job_id}: {str(e)}")

        return jobs

    @staticmethod
    def _clean_html_content(html_content: str) -> str:
        """
        Clean HTML content from job descriptions
        - Remove HTML tags while preserving important formatting
        - Convert lists to plain text with proper formatting
        - Preserve important line breaks and structure
        """
        if not html_content:
            return ""

        try:
            # Check if content is HTML
            if "<" in html_content and ">" in html_content:
                soup = BeautifulSoup(html_content, "html.parser")

                # Replace <br> with newlines
                for br in soup.find_all("br"):
                    br.replace_with("\n")

                # Convert lists to formatted text
                for ul in soup.find_all("ul"):
                    for li in ul.find_all("li"):
                        li_text = li.get_text().strip()
                        li.replace_with(f"• {li_text}\n")

                for ol in soup.find_all("ol"):
                    for i, li in enumerate(ol.find_all("li"), 1):
                        li_text = li.get_text().strip()
                        li.replace_with(f"{i}. {li_text}\n")

                # Handle paragraphs
                for p in soup.find_all("p"):
                    p_text = p.get_text().strip()
                    p.replace_with(f"{p_text}\n\n")

                # Handle headings
                for i in range(1, 7):
                    for h in soup.find_all(f"h{i}"):
                        h_text = h.get_text().strip()
                        h.replace_with(f"{h_text}\n")

                # Get the cleaned text
                text = soup.get_text()

                # Final cleanup
                text = re.sub(r"\n{3,}", "\n\n", text)  # Replace multiple newlines
                text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces

                return text.strip()
            else:
                # If not HTML, just return as is
                return html_content.strip()

        except Exception as e:
            logger.error(f"Error cleaning HTML content: {str(e)}")
            # Fallback to simple HTML stripping
            return re.sub(r"<[^>]+>", " ", html_content).strip()
