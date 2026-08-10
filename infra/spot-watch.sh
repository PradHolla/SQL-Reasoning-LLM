#!/usr/bin/env bash
# Watch for a spot interruption notice and rescue work to S3 before the
# instance disappears.
#
# AWS gives a two-minute warning via the instance metadata service. That is
# enough to sync checkpoints, and not enough to do anything clever — so this
# does exactly one thing, fast.
#
# Installed by user-data.sh as a systemd service.

BUCKET="${SQL_LLM_BUCKET:?SQL_LLM_BUCKET must be set}"
WATCH_DIR="${WATCH_DIR:-/opt/sql-llm/repo/outputs}"
IMDS="http://169.254.169.254/latest"

log() { logger -t spot-watch "$*"; echo "[spot-watch] $*"; }

log "watching for interruption notices; will rescue ${WATCH_DIR} -> s3://${BUCKET}/outputs/"

while true; do
    TOKEN=$(curl -sX PUT "${IMDS}/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null)

    CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "X-aws-ec2-metadata-token: ${TOKEN}" \
        "${IMDS}/meta-data/spot/instance-action" 2>/dev/null)

    # 200 = interruption scheduled. 404 = normal, nothing happening.
    if [ "$CODE" = "200" ]; then
        log "INTERRUPTION NOTICE — syncing checkpoints to S3"
        aws s3 sync "$WATCH_DIR" "s3://${BUCKET}/outputs/" --only-show-errors
        log "sync complete; instance will terminate shortly"
        exit 0
    fi

    sleep 5
done
