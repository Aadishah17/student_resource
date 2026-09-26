# AWS Deployment

## 1. Source Code Deployment

The Business Entity Resolution project was cloned from GitHub onto the Amazon EC2 instance.

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

The combined test dataset was approximately 1.2 GB.

## 3. Model Deployment

The trained model was included in the repository.

Model directory:

`code/business_entity_resolution/models/`

The deployed inference configuration used:

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

The environment contained the required inference dependencies, including:

- pandas
- NumPy
- XGBoost
- RapidFuzz
- psutil

Using the virtual environment avoided installing project dependencies into the system Python environment.

## 6. Persistent Execution

Long-running inference was executed inside a `tmux` session.

Example:

```bash
tmux new -s inference
````

The inference process could continue running independently of the SSH connection.

The session could be reattached using:

```bash
tmux attach -t inference
```

## 7. Checkpointing and Resume

The inference pipeline supports checkpoint-based recovery.

The checkpoint records:

* Completed batches
* Number of S1 entities written
* Matching output byte offset
* Candidate output byte offset
* Last completed country
* Last completed batch

This allows interrupted inference to resume without duplicating completed output.

An example checkpoint from the AWS smoke run recorded the completed France partition:

`France_batch_0_1490`

## 8. Initial Memory Issue

The initial AWS inference attempt used the default batch size of `50,000`.

The process exceeded the available physical memory and was terminated by the Linux OOM killer.

The EC2 instance had:

* 2 vCPUs
* 8 GiB RAM
* No swap initially

## 9. Memory Optimization

The AWS deployment was subsequently adjusted by:

1. Creating an 8 GiB swap file.
2. Reducing the inference batch size from `50,000` to `10,000`.
3. Running inference inside `tmux`.

The resulting configuration was:

```text
Physical RAM: 8 GiB
Swap: 8 GiB
Batch size: 10,000
Threshold: 0.72
```

This configuration successfully completed the AWS smoke test.

## 10. AWS Smoke Test

The successful smoke test processed 10,000 S1 entities across the available country partitions:

| Country   | S1 entities |
| --------- | ----------: |
| France    |       1,490 |
| India     |       4,619 |
| US        |       3,891 |
| **Total** |  **10,000** |

The inference completed successfully.

The highest observed RAM usage during scoring was approximately `7.36 GB`.

## 11. Smoke-Test Outputs

The successful smoke-test outputs were preserved separately from the clean production output directory.

Location:

`output_smoke_10k/`

Files:

* `matching_results.tsv`
* `candidate_pairs.tsv`

The production output directory was kept separate:

`output/`

This prevented the smoke-test results from being mixed with a future full inference.

## 12. Validation

The official challenge validator was executed after the smoke test.

The validator confirmed that the generated files were structurally processed, but reported missing S1 entities because the smoke test intentionally processed only 10,000 entities out of the complete test set.

The test set contains approximately 1.73 million S1 entities.

Therefore, the validator result from this run represents a **smoke-test validation**, not a final submission validation.

## 13. Deployment Outcome

The AWS deployment successfully demonstrated that the entity-resolution inference pipeline could:

1. Run on an Ubuntu EC2 environment.
2. Access the challenge dataset through Amazon S3 using an IAM instance role.
3. Load the trained XGBoost model.
4. Perform large-scale candidate generation and scoring.
5. Process multiple country partitions.
6. Persist inference checkpoints.
7. Resume interrupted work.
8. Produce matching and candidate output files.
9. Execute the official challenge validator.

The complete 1.73-million-S1 production inference was not completed during this AWS deployment period.

```