# AWS Deployment

## 1. Source Code Deployment

The Business Entity Resolution project was cloned from GitHub onto the EC2 instance.
Repository location on EC2:

`~/student_resource`

The deployment included:

- Entity-resolution source code
- Trained XGBoost model
- Model metadata
- Inference scripts
- Validation utilities
- Tests
- Requirements

## 2. Dataset Deployment

The challenge test dataset was stored in Amazon S3 and downloaded to the EC2 instance.

Local dataset location:

`~/student_resource/dataset/test/`

The inference pipeline used:

- `test_source1.tsv`
- `test_source2.tsv`
- `test_source3.tsv`

## 3. Model Deployment

The trained model was already included in the repository.

Model directory:

`code/business_entity_resolution/models/`

The production inference configuration used:

- XGBoost model
- 96 engineered features
- Configuration H blocking
- Decision threshold: `0.72`

## 4. Inference Runner

The AWS inference entry point was:

`run_aws_inference.sh`

The runner supports configurable:

- Data directory
- Model directory
- Output directory
- Batch size
- Prediction threshold
- Python executable
- Resume/checkpoint behavior
- Smoke and full inference modes

## 5. Python Environment

The inference process used the project virtual environment:

`.venv/bin/python`

This avoided installing project dependencies into the system Python environment.

## 6. Persistent Execution

Long-running inference was executed inside a `tmux` session.

Example:

```text
tmux new -s inference

