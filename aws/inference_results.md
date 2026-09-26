# AWS Inference Results

## AWS Environment

| Parameter | Value |
|---|---|
| Region | `ap-south-1` |
| Instance | `m7i-flex.large` |
| CPU | 2 vCPU |
| RAM | 8 GiB |
| Swap | 8 GiB |
| Storage | 100 GiB gp3 |
| OS | Ubuntu Server 24.04 LTS |
| Batch size | 10,000 |
| Threshold | 0.72 |
| Model | XGBoost |
| Features | 96 |

## Smoke Test

A 10,000-S1 smoke test was executed on the AWS EC2 instance.

### Dataset Distribution

| Country | S1 Entities |
|---|---:|
| France | 1,490 |
| India | 4,619 |
| US | 3,891 |
| **Total** | **10,000** |

## Processing Results

The inference pipeline completed successfully across all three country partitions.

The final smoke-test output contained:

- `10,000` processed S1 entities
- `3,949` predicted non-empty matches
- `6,051` predicted empty matches
- `candidate_pairs.tsv`
- `matching_results.tsv`

## Candidate Generation

The smoke test processed candidate pairs before model scoring.

For the US partition:

- S1 entities: `3,891`
- Candidate pairs: `557,059`
- Unique targets: `201,797`

## Runtime

The complete smoke-test pipeline reported:

`52.51 minutes`

The US partition alone completed in approximately:

`315.91 seconds`

## Memory Usage

The highest observed memory usage during the successful run was approximately:

`7.36 GB`

The initial configuration without swap resulted in an OOM termination.

The successful configuration used:

- 8 GiB physical RAM
- 8 GiB swap
- Batch size of 10,000

## Output Files

The successful smoke-test outputs were preserved in:

`output_smoke_10k/`

Files:

- `matching_results.tsv`
- `candidate_pairs.tsv`

Approximate sizes:

| File | Size |
|---|---:|
| `matching_results.tsv` | 254 KB |
| `candidate_pairs.tsv` | 18 MB |

## Validation

The official `validate_submission.py` validator was executed.

The validator reported missing S1 entities because only 10,000 S1 entities were processed while the complete test set contains approximately 1.73 million S1 entities.

This was expected because the execution was a smoke test rather than a complete submission run.

The validation result therefore should not be interpreted as the final challenge submission result.

## AWS Deployment Outcome

The AWS deployment successfully demonstrated:

- S3-based dataset delivery
- IAM-based EC2-to-S3 access
- EC2-based inference execution
- XGBoost model deployment
- Multi-country processing
- Candidate generation
- Batch inference
- Checkpointing
- Resume capability
- Output generation
- Official validator execution

A complete 1.73-million-S1 production inference was not completed during this deployment period.
