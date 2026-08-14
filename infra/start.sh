#!/usr/bin/env bash
# Restart a stopped project instance and wait until it is usable.
#
#   ./infra/start.sh                 start the stopped sql-reasoning-llm box
#   ./infra/start.sh i-0abc...       start one
#   ./infra/start.sh --no-pull       skip the git pull
#
# Use this rather than launch.sh after an idle shutdown. A stopped instance
# keeps its root volume, so the repo, the uv environment and the ~2GB HuggingFace
# cache are all still there — relaunching would download all of it again.
#
# The public IP changes on every start (there is no Elastic IP), so this prints
# the new SSH command and refreshes the security group rule if your address has
# also moved.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

PULL=1
IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-pull) PULL=0; shift ;;
        i-*) IDS+=("$1"); shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#IDS[@]} -eq 0 ]]; then
    # No mapfile: macOS ships bash 3.2, which predates it.
    FOUND=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=sql-reasoning-llm" \
                  "Name=instance-state-name,Values=stopped,stopping" \
        --query 'Reservations[].Instances[].InstanceId' --output text)
    for ID in $FOUND; do IDS+=("$ID"); done
fi

if [[ ${#IDS[@]} -eq 0 || -z "${IDS[0]:-}" ]]; then
    echo "no stopped project instances — is one already running?"
    echo "  ./infra/status.sh"
    exit 0
fi

echo "==> starting: ${IDS[*]}"
aws ec2 start-instances --instance-ids "${IDS[@]}" \
    --query 'StartingInstances[].{Id:InstanceId,State:CurrentState.Name}' --output table

aws ec2 wait instance-running --instance-ids "${IDS[@]}"

# --- keep SSH reachable ---------------------------------------------------
# The SG allows one address. Home IPs move; re-authorising is idempotent enough
# to just try and ignore the duplicate error.
MYIP=$(curl -s --max-time 10 https://checkip.amazonaws.com | tr -d '\n' || true)
if [[ -n "$MYIP" ]]; then
    if aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr "${MYIP}/32" >/dev/null 2>&1; then
        echo "==> added SSH rule for ${MYIP}/32"
    fi
fi

for ID in "${IDS[@]}"; do
    IP=$(aws ec2 describe-instances --instance-ids "$ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

    echo "==> $ID up at $IP — waiting for sshd..."
    for _ in $(seq 1 30); do
        if ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
               "ubuntu@$IP" true 2>/dev/null; then
            break
        fi
        sleep 5
    done

    if [[ "$PULL" == "1" ]]; then
        # results/ is discarded before pulling, every time. Eval runs write
        # there, so the box always has modified and untracked files under it,
        # and the pull aborts. That has now blocked a pull four separate times,
        # and twice the failure was masked because the calling command piped
        # git through `tail`. The box's copies are never the only copies: the
        # workflow is to scp results down, commit them on the laptop, and push.
        #
        # Scoped to results/ deliberately. `git checkout -- .` would silently
        # destroy a real fix someone made on the box at 2am.
        echo "==> git pull"
        ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "ubuntu@$IP" '
            cd /opt/sql-llm/repo &&
            git checkout -- results/ 2>/dev/null;
            git clean -qfd results/ 2>/dev/null;
            git pull --ff-only && git log --oneline -1
        ' || echo "  (pull failed — uncommitted work on the box outside results/?)"
    fi

    cat <<EOF

  instance : $ID
  ip       : $IP          (new on every start — no Elastic IP)
  ssh      : ssh -i $KEY_FILE ubuntu@$IP

  Stop when done:   ./infra/stop.sh $ID
  Idle shutdown:    after ${IDLE_SHUTDOWN_MINUTES}m below ${IDLE_GPU_PCT}% GPU
EOF
done
