# AWS Architecture

## Overview

The AWS deployment separated source code, compute, data storage, and permissions.

## Architecture

GitHub
  |
  | git clone
  v
Amazon EC2
  |
  | IAM Instance Role
  v
Amazon S3

## Components

### GitHub

The repository contains the source code, trained model, inference scripts, tests, utilities, and documentation.

### Amazon EC2

EC2 was used as the compute environment for large-scale entity-resolution inference.

Configuration:

- Instance type: m7i-flex.large
- CPU: 2 vCPU
- Memory: 8 GiB
- Operating system: Ubuntu Server 24.04 LTS
- Storage: 100 GiB gp3

### Amazon S3

S3 was used to store the challenge test dataset.

Bucket:

`amazon-ml-2026-nitya`

Dataset path:

`s3://amazon-ml-2026-nitya/dataset/test/`

### AWS IAM

The EC2 instance used the IAM role:

`amazon-ml-2026-ec2-s3-role`

The role provided EC2 access to the required S3 resources without storing AWS access keys on the server.

## Region

AWS region:

`ap-south-1` — Mumbai
