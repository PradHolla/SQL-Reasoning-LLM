#!/usr/bin/env bash
# Stop (or terminate) project instances.
#
#   ./infra/stop.sh                 stop every running sql-reasoning-llm box
#   ./infra/stop.sh i-0abc...       stop one
#   ./infra/stop.sh --terminate     terminate instead of stop (destroys the disk)
#
# Stop keeps the root volume, so your work survives and you pay only for EBS
# (~$16/mo for a 200GB gp3). Terminate destroys everything not synced to S3.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

ACTION="stop-instances"
VERB="stopping"
IDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --terminate) ACTION="terminate-instances"; VERB="TERMINATING"; shift ;;
        i-*) IDS+=("$1"); shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#IDS[@]} -eq 0 ]]; then
    # No mapfile here on purpose: macOS ships bash 3.2, which predates it.
    FOUND=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=sql-reasoning-llm" \
                  "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[].InstanceId' --output text)
    for ID in $FOUND; do IDS+=("$ID"); done
fi

if [[ ${#IDS[@]} -eq 0 || -z "${IDS[0]:-}" ]]; then
    echo "no running project instances"
    exit 0
fi

echo "==> $VERB: ${IDS[*]}"
if [[ "$ACTION" == "terminate-instances" ]]; then
    read -rp "This destroys the root volume. Type 'yes' to confirm: " CONFIRM
    [[ "$CONFIRM" == "yes" ]] || { echo "aborted"; exit 1; }
fi

if [[ "$ACTION" == "terminate-instances" ]]; then
    QUERY='TerminatingInstances[].{Id:InstanceId,State:CurrentState.Name}'
else
    QUERY='StoppingInstances[].{Id:InstanceId,State:CurrentState.Name}'
fi

aws ec2 "$ACTION" --instance-ids "${IDS[@]}" --query "$QUERY" --output table
