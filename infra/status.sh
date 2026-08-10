#!/usr/bin/env bash
# What is running, what is it costing, and is the quota through yet.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

echo "=== project instances ($AWS_REGION) ==="
aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=sql-reasoning-llm" \
    --query 'Reservations[].Instances[?State.Name!=`terminated`].{Id:InstanceId,Type:InstanceType,State:State.Name,IP:PublicIpAddress,Life:InstanceLifecycle,Launched:LaunchTime}' \
    --output table

echo "=== G/VT spot quota (0 = spot launches will fail) ==="
aws service-quotas get-service-quota --service-code ec2 --quota-code L-3819A6DF \
    --query 'Quota.Value' --output text

# Statuses seen in the wild: PENDING (auto-review), CASE_OPENED (a human is
# looking at it), APPROVED, DENIED. Don't filter — an empty list here would
# otherwise look identical to "no request was ever filed".
echo "=== quota request history ==="
aws service-quotas list-requested-service-quota-change-history \
    --service-code ec2 \
    --query 'RequestedQuotas[].{Quota:QuotaName,Desired:DesiredValue,Status:Status,Created:Created}' \
    --output table

echo "=== month-to-date spend vs \$100 budget ==="
aws budgets describe-budget --account-id "$ACCOUNT_ID" --budget-name sql-reasoning-llm-ec2 \
    --query 'Budget.{Limit:BudgetLimit.Amount,Spent:CalculatedSpend.ActualSpend.Amount,Forecast:CalculatedSpend.ForecastedSpend.Amount}' \
    --output table 2>/dev/null || echo "(budget data lags a few hours)"

echo "=== current spot prices, $INSTANCE_TYPE ==="
aws ec2 describe-spot-price-history \
    --instance-types "$INSTANCE_TYPE" --product-descriptions "Linux/UNIX" \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
    --query 'sort_by(SpotPriceHistory,&to_number(SpotPrice))[].{AZ:AvailabilityZone,Price:SpotPrice}' \
    --output table
