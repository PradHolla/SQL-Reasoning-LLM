#!/usr/bin/env bash
# Shared configuration for all infra scripts. Sourced, not executed.
#
# Everything here was provisioned once; see infra/README.md for what exists
# and how to recreate it.

export AWS_REGION="${AWS_REGION:-us-east-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"

# --- account resources ---------------------------------------------------
ACCOUNT_ID="381492233412"
BUCKET="sql-reasoning-llm-381492233412-us-east-2"
KEY_NAME="sql-llm-ue2"
KEY_FILE="${KEY_FILE:-$HOME/.ssh/sql-llm-ue2.pem}"
SG_ID="sg-0c9d9c853ea1d03b7"
IAM_PROFILE="sql-llm-ec2-profile"
VPC_ID="vpc-050a6be2968570ada"
REPO_URL="https://github.com/PradHolla/SQL-Reasoning-LLM.git"

# --- instance defaults ---------------------------------------------------
# g5.xlarge = 1x A10G 24GB, 4 vCPU. Enough through Phase 2.
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-200}"

# Base NVIDIA-driver AMI. PyTorch is NOT preinstalled on purpose — uv installs
# its own, and pip torch wheels bundle the CUDA runtime. Only the driver has to
# come from the AMI.
AMI_SSM_PARAM="/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"

# --- guardrails ----------------------------------------------------------
# Minutes of sustained GPU idleness before the box shuts itself down.
IDLE_SHUTDOWN_MINUTES="${IDLE_SHUTDOWN_MINUTES:-30}"
IDLE_GPU_PCT="${IDLE_GPU_PCT:-5}"

export ACCOUNT_ID BUCKET KEY_NAME KEY_FILE SG_ID IAM_PROFILE VPC_ID REPO_URL
export INSTANCE_TYPE VOLUME_SIZE_GB AMI_SSM_PARAM
export IDLE_SHUTDOWN_MINUTES IDLE_GPU_PCT
