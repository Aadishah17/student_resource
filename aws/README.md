# AWS Deployment

This directory documents the AWS infrastructure and deployment used for the Business Entity Resolution Challenge.

## Purpose

AWS was used to execute large-scale entity-resolution inference against the challenge test dataset.

## Architecture

The deployment used:

- GitHub for source code
- Amazon EC2 for inference
- Amazon S3 for test-data storage
- AWS IAM for EC2-to-S3 permissions

See [architecture.md](architecture.md) for the architecture and [deployment.md](deployment.md) for the execution workflow.

## AWS Region

`ap-south-1` — Mumbai

## Infrastructure

- EC2 instance: `m7i-flex.large`
- vCPUs: 2
- Memory: 8 GiB
- Storage: 100 GiB gp3
- Operating system: Ubuntu Server 24.04 LTS
- S3 bucket: `amazon-ml-2026-nitya`
- IAM role: `amazon-ml-2026-ec2-s3-role`

## Inference Configuration

- Model: XGBoost
- Features: 96
- Decision threshold: 0.72
- Smoke-test batch size: 10,000 S1 entities
- Checkpoint/resume support: enabled
- Persistent terminal session: tmux

## Result

A 10,000-S1 smoke inference completed successfully on EC2.

Peak observed RAM usage was approximately 7.36 GB. An 8 GB swap file was subsequently configured to provide additional memory safety.
