from jobs.base import Job

class Training(Job):
    def execute(self, step, config, logger):
        print(f"[Training] Executing {step.__class__.__name__}")
        step.process(config, logger)
