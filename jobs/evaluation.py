from jobs.base import Job

class Evaluation(Job):
    def execute(self, step):
        print(f"[Evaluation] Executing {step.__class__.__name__}")
        step.process()
