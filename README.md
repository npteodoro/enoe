# Enoe: Modular Job Execution Framework

Enoe is a modular framework designed to execute various jobs (e.g., training, evaluation) with specific steps (e.g., classification, forecasting, segmentation). It leverages object-oriented principles, dependency injection, and YAML-based configuration for flexibility and extensibility.

## Features

* Modular Job Execution: Easily execute different jobs and steps
* Flexible Configuration: YAML-based configuration files for easy adjustments
* Logging with TensorBoard: Integrated logging for easy monitoring and debugging
* Dependency Injection: Pass configuration and logger objects downstream efficiently
* Extensible Architecture: Add new jobs and steps with minimal effort

## Project Structure

```text
enoe/
├── enoe.py                     # Main entry point
├── configs/
│   └── config.yaml             # YAML configuration file
├── factories/
│   ├── __init__.py
│   ├── job_executor.py         # Executes jobs and steps
│   ├── job_factory.py          # Creates job instances
│   └── step_factory.py         # Creates step instances
├── training/
│   ├── __init__.py
│   ├── classification.py       # Classification training step
│   ├── forecasting.py          # Forecasting training step
│   └── segmentation.py         # Segmentation training step
├── evaluation/
│   ├── __init__.py
│   ├── classification.py       # Classification evaluation step
│   ├── forecasting.py          # Forecasting evaluation step
│   └── segmentation.py         # Segmentation evaluation step
├── utils/
│   ├── __init__.py
│   ├── config.py               # Configuration loader class
│   └── logger.py               # Logger class (TensorBoard)
└── README.md                   # Project documentation
```

# Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ynpteodoro/enoe.git
cd enoe
pip install -r requirements.txt
```

# Usage

Execute a job with a specific step:

```bash
python enoe.py training segmentation
```

# Configuration

Configuration is managed via YAML files (configs/config.yaml):

```yaml
model:
  encoder_name: "resnet50"
  encoder_weights: "imagenet"
  in_channels: 3
  classes: 10

training:
  batch_size: 16
  num_epochs: 10
  learning_rate: 0.001
  num_workers: 4
```

# Logging

Logs are stored in TensorBoard format. To view logs:

```bash
tensorboard --logdir=logs
```

# Extending the Framework

Adding a New Job or Step

1. Create a new Python file in the appropriate directory (`training` or `evaluation`)
1. Implement the required methods (`execute` for jobs, `process` for steps)
1. Register the new job or step in the corresponding factory (`JobFactory` or `StepFactory`)

# Contributing

Contributions are welcome! Please fork the repository, create a branch, and submit a pull request.

# License

This project is licensed under the MIT License.
