#!/usr/bin/env bash
# Launch a GPU box from zero to ready, in one command.
#
#   ./infra/launch.sh                      on-demand g5.xlarge
#   ./infra/launch.sh --spot               spot (needs the G/VT spot quota)
#   ./infra/launch.sh --type g5.2xlarge    bigger box
#
# Prints the SSH command when the instance is running. Bootstrap continues in
# the background for a few minutes after that — see `status.sh`.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

USE_SPOT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --spot)   USE_SPOT=1; shift ;;
        --type)   INSTANCE_TYPE="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "==> region=$AWS_REGION type=$INSTANCE_TYPE spot=$USE_SPOT"

# --- resolve the current AMI ---------------------------------------------
AMI_ID=$(aws ssm get-parameter --name "$AMI_SSM_PARAM" --query 'Parameter.Value' --output text)
echo "==> AMI $AMI_ID"

ROOT_DEV=$(aws ec2 describe-images --image-ids "$AMI_ID" \
    --query 'Images[0].RootDeviceName' --output text)

# --- pick a subnet --------------------------------------------------------
# For spot, choose the AZ with the cheapest current price. Spot pricing varies
# a lot by AZ and this is free money.
if [[ "$USE_SPOT" == "1" ]]; then
    CHEAPEST_AZ=$(aws ec2 describe-spot-price-history \
        --instance-types "$INSTANCE_TYPE" \
        --product-descriptions "Linux/UNIX" \
        --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
        --query 'sort_by(SpotPriceHistory,&to_number(SpotPrice))[0].AvailabilityZone' \
        --output text)
    PRICE=$(aws ec2 describe-spot-price-history \
        --instance-types "$INSTANCE_TYPE" \
        --product-descriptions "Linux/UNIX" \
        --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
        --query 'sort_by(SpotPriceHistory,&to_number(SpotPrice))[0].SpotPrice' \
        --output text)
    echo "==> cheapest AZ $CHEAPEST_AZ at \$$PRICE/hr"
    SUBNET_ID=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$CHEAPEST_AZ" \
        --query 'Subnets[0].SubnetId' --output text)
else
    SUBNET_ID=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
        --query 'Subnets[0].SubnetId' --output text)
fi
echo "==> subnet $SUBNET_ID"

# --- render user-data -----------------------------------------------------
UD=$(mktemp)
trap 'rm -f "$UD"' EXIT
sed -e "s|__REGION__|$AWS_REGION|g" \
    -e "s|__BUCKET__|$BUCKET|g" \
    -e "s|__REPO_URL__|$REPO_URL|g" \
    -e "s|__IDLE_SHUTDOWN_MINUTES__|$IDLE_SHUTDOWN_MINUTES|g" \
    -e "s|__IDLE_GPU_PCT__|$IDLE_GPU_PCT|g" \
    "$HERE/user-data.sh" > "$UD"

# --- market options -------------------------------------------------------
# Spot must terminate on shutdown; on-demand stops instead, so the root volume
# and everything on it survives an idle shutdown.
MARKET_ARGS=()
SHUTDOWN_BEHAVIOR="stop"
if [[ "$USE_SPOT" == "1" ]]; then
    MARKET_ARGS=(--instance-market-options \
        'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
    SHUTDOWN_BEHAVIOR="terminate"
fi

# --- launch ---------------------------------------------------------------
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-initiated-shutdown-behavior "$SHUTDOWN_BEHAVIOR" \
    --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
    --user-data "file://$UD" \
    --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=sql-llm-gpu},{Key=Project,Value=sql-reasoning-llm}]" \
        "ResourceType=volume,Tags=[{Key=Project,Value=sql-reasoning-llm}]" \
    ${MARKET_ARGS[@]+"${MARKET_ARGS[@]}"} \
    --query 'Instances[0].InstanceId' --output text)

echo "==> launched $INSTANCE_ID — waiting for running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<EOF

  instance : $INSTANCE_ID
  ip       : $IP
  ssh      : ssh -i $KEY_FILE ubuntu@$IP

Bootstrap (driver check, clone, uv sync) runs for a few more minutes.
Watch it with:
  ssh -i $KEY_FILE ubuntu@$IP 'tail -f /var/log/user-data.log'

Ready when /opt/sql-llm/.bootstrap-complete exists.

Stop it when done:      ./infra/stop.sh $INSTANCE_ID
Idle auto-shutdown:     after ${IDLE_SHUTDOWN_MINUTES}m below ${IDLE_GPU_PCT}% GPU
EOF
