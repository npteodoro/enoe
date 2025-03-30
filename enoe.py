import argparse
from factories.job_executor import JobExecutor
from utils.config import ConfigLoader
from utils.logger import Logger

def parse_arguments():
    # Define valid options for job and step
    valid_jobs = ["training", "evaluation"]
    valid_steps = ["classification", "forecasting", "segmentation"]

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Execute a job with a specific step.")
    parser.add_argument(
        "job",
        type=str,
        choices=valid_jobs,
        help=f"The type of job to execute. Valid options: {', '.join(valid_jobs)}."
    )
    parser.add_argument(
        "step",
        type=str,
        choices=valid_steps,
        help=f"The type of step to execute. Valid options: {', '.join(valid_steps)}."
    )

    # Parse arguments
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config = ConfigLoader(job=args.job, step=args.step)
    if config is None:
        print("Error: Configuration loading failed.")
        return

    # Initialize logger
    logger = Logger(config=config)

    # Use JobExecutor to execute the job
    executor = JobExecutor(config, logger)
    executor.execute(args.job, args.step)

    logger.close()

if __name__ == "__main__":
    main()