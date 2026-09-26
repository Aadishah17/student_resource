# AWS Setup

## 1. AWS Region

The deployment was created in:

`ap-south-1` — Mumbai

## 2. Amazon S3

An S3 bucket was created for the challenge dataset:

`amazon-ml-2026-nitya`

The test dataset was stored at:

`s3://amazon-ml-2026-nitya/dataset/test/`

Files:

- `test_source1.tsv`
- `test_source2.tsv`
- `test_source3.tsv`

The combined dataset size was approximately 1.2 GB.

## 3. AWS IAM

An IAM role was created for the EC2 instance:

`amazon-ml-2026-ec2-s3-role`

The role allowed the EC2 instance to access the required S3 resources without storing AWS access keys on the server.

Permissions used included:

- `s3:ListBucket`
- `s3:GetObject`
- `s3:PutObject`

## 4. Amazon EC2

The inference environment used:

- Instance type: `m7i-flex.large`
- vCPUs: 2
- Memory: 8 GiB
- Operating system: Ubuntu Server 24.04 LTS
- Storage: 100 GiB gp3
- Region: `ap-south-1`
- Availability Zone: `ap-south-1a`

## 5. SSH Access

The EC2 instance was accessed using the existing EC2 key pair:

`amazon-ml-2026-key`

SSH access was restricted using the configured security group.

## 6. AWS CLI

AWS CLI v2 was installed on the EC2 instance to allow interaction with the S3 bucket.

The EC2 IAM role was used for authentication.

## 7. Repository Deployment

The GitHub repository was cloned onto EC2.

A sparse checkout was used to avoid unnecessarily downloading large repository content.

The project was located at:

`~/student_resource`

## 8. Python Environment

Python 3.12 was available on the Ubuntu instance.

A project virtual environment was created:

`.venv`

The inference environment included:

- pandas
- NumPy
- XGBoost
- RapidFuzz
- psutil

## 9. Test Dataset

The test dataset was downloaded from S3 to:

`~/student_resource/dataset/test/`

The three TSV files were used directly by the inference pipeline.

## 10. Memory Configuration

The initial inference attempt using a batch size of 50,000 exceeded the available physical memory and was terminated by the Linux OOM killer.

An 8 GiB swap file was subsequently configured:

`/swapfile`

The inference batch size was reduced to:

`10,000`

This configuration allowed the 10,000-S1 smoke test to complete successfully.

## 11. Persistent Inference Session

The inference process was executed inside `tmux`.

This allowed the process to continue running independently of the SSH connection.

## 12. Checkpoint and Resume

The inference pipeline supports checkpointing and resume.

Completed batches and output offsets are recorded so that an interrupted inference can continue without duplicating completed output.
