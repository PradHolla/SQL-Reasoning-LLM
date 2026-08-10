#!/usr/bin/env bash
# Shut the instance down after sustained GPU idleness.
#
# Installed by user-data.sh as a root cron job running every minute. This is the
# single highest-value cost control in the project: a forgotten g5.xlarge is
# ~$170/week, and this script caps that at the idle threshold.
#
# To prevent shutdown during long CPU-only work:  touch /opt/sql-llm/.no-autoshutdown

THRESHOLD_PCT="${IDLE_GPU_PCT:-5}"
IDLE_MINUTES="${IDLE_SHUTDOWN_MINUTES:-30}"
STATE=/var/run/gpu-idle-count
HOLD=/opt/sql-llm/.no-autoshutdown

# Manual hold — reset the counter and do nothing.
if [ -f "$HOLD" ]; then
    echo 0 > "$STATE"
    exit 0
fi

# Highest utilization across all GPUs. If nvidia-smi is missing or the driver
# isn't up yet, do nothing rather than risk shutting down a healthy box.
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
if [ -z "$UTIL" ]; then
    exit 0
fi

COUNT=$(cat "$STATE" 2>/dev/null || echo 0)
if [ "$UTIL" -lt "$THRESHOLD_PCT" ]; then
    COUNT=$((COUNT + 1))
else
    COUNT=0
fi
echo "$COUNT" > "$STATE"

if [ "$COUNT" -ge "$IDLE_MINUTES" ]; then
    logger -t idle-shutdown "GPU idle ${IDLE_MINUTES}m (util ${UTIL}%) — shutting down"
    wall "GPU has been idle ${IDLE_MINUTES} minutes. Shutting down. Prevent with: touch ${HOLD}" 2>/dev/null
    sleep 10
    /sbin/shutdown -h now
fi
