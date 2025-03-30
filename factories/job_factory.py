from jobs.training import Training
from jobs.evaluation import Evaluation

class JobFactory:
    """Factory class to create Job instances based on job_type."""
    @staticmethod
    def create_job(job_type: str):
        jobs = {
            "training": Training,
            "evaluation": Evaluation
        }
        job = jobs.get(job_type.lower())
        if not job:
            print(f"Error: Invalid job type '{job_type}'")
            return None
        return job()
