from factories.job_factory import JobFactory
from factories.step_factory import StepFactory

class JobExecutor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def execute(self, job_type: str, step_type: str):
        job = JobFactory.create_job(job_type)
        step = StepFactory.create_step(job_type, step_type, self.config, self.logger)

        if job and step:
            self.logger.add_text("Job Execution", f"Executing {job_type} -> {step_type}")
            job.execute(step)
        else:
            self.logger.add_text("Error", "Error: Job or Step invalid")
            print("Error: Job or Step invalid")
