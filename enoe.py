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
    parser.add_argument(
        "--encoder-name",
        type=str,
        default=None,
        help="The name of the encoder to use. Default is 'default_encoder'."
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="The architecture to use. Default is configs/config.yaml."
    )

    # Parse arguments
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config = ConfigLoader()
    config.load(job=args.job, step=args.step)
    if args.encoder_name:
        config.set_encoder_name(args.encoder_name)
    if args.architecture:
        config.set_model_architecture(args.architecture)

    # Initialize logger
    logger = Logger(config=config)

    # Use JobExecutor to execute the job
    executor = JobExecutor(config, logger)
    executor.execute(args.job, args.step)

    logger.close()

if __name__ == "__main__":
    main()
