#!/usr/bin/env bash
# Connect to the running GPU box without looking up its IP.
#
#   ./infra/ssh.sh                    interactive shell, in the repo
#   ./infra/ssh.sh 'nvidia-smi'       run one command and exit
#   ./infra/ssh.sh --log              follow the current eval run
#   ./infra/ssh.sh --gpu              live GPU utilisation
#   ./infra/ssh.sh --ip               just print the IP
#
# The public IP changes every time the instance starts, so hardcoding it in
# notes or shell history does not work. This resolves it from the Project tag.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

IP=$(aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=sql-reasoning-llm" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

if [[ -z "$IP" || "$IP" == "None" ]]; then
    echo "no running project instance." >&2
    echo "  ./infra/start.sh     restart a stopped one" >&2
    echo "  ./infra/launch.sh    create a new one" >&2
    exit 1
fi

SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "ubuntu@$IP")

case "${1:-}" in
    --ip)
        echo "$IP"
        ;;
    --log)
        # The raw log is mostly model-loading progress bars. Show the parts a
        # human wants: which model is running, throughput, and the scores.
        "${SSH[@]}" 'tail -f -n 200 /tmp/eval_test.log | grep --line-buffered -aE \
            "^########|generated |loaded on |rate |accuracy|match|^model |^  [a-z-]+ +[0-9]"'
        ;;
    --gpu)
        "${SSH[@]}" 'watch -n 2 nvidia-smi'
        ;;
    "")
        "${SSH[@]}" -t 'cd /opt/sql-llm/repo && exec bash -l'
        ;;
    *)
        "${SSH[@]}" "cd /opt/sql-llm/repo && $*"
        ;;
esac
