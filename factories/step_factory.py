from training.classification import TrainingClassification
from training.forecasting import TrainingForecasting
from training.segmentation import TrainingSegmentation
from evaluation.classification import EvaluationClassification
from evaluation.forecasting import EvaluationForecasting
from evaluation.segmentation import EvaluationSegmentation
from utils.config import ConfigLoader
from utils.logger import Logger

class StepFactory:
    """Factory class to create Step instances based on job_type and step_type."""

    @staticmethod
    def create_step(job_type: str, step_type: str, config: ConfigLoader, logger: Logger):
        steps = {
            "training": {
                "classification": TrainingClassification,
                "forecasting": TrainingForecasting,
                "segmentation": TrainingSegmentation
            },
            "evaluation": {
                "classification": EvaluationClassification,
                "forecasting": EvaluationForecasting,
                "segmentation": EvaluationSegmentation
            }
        }
        step = steps.get(job_type.lower(), {}).get(step_type.lower())
        if not step:
            print(f"Error: Invalid step type '{step_type}' for job type '{job_type}'")
            return None
        return step(config, logger)
