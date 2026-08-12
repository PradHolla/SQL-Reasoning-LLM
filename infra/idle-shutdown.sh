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

# GPU utilization alone is not "is this box in use". An evaluation run spends
# real time loading weights, building prompts, executing SQL and scoring, all at
# 0% GPU — and this script shut down a live run twice before that was fixed.
#
# So idleness is graded. A process still holding GPU memory, or a human still
# logged in, means work is probably in flight: those get a much longer rope
# rather than immunity, because "forgot to log out" must not cost $170/week.
BUSY_MINUTES="${IDLE_BUSY_MINUTES:-180}"

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

# Is anything still holding the GPU? A process between batches, or in a long
# CPU-only phase of a job, reads 0% but has memory allocated.
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0)
# Is a human still connected?
SESSIONS=$(who 2>/dev/null | grep -c . || echo 0)

if [ "$GPU_PROCS" -gt 0 ] || [ "$SESSIONS" -gt 0 ]; then
    LIMIT="$BUSY_MINUTES"
    REASON="in use (gpu procs=${GPU_PROCS}, sessions=${SESSIONS})"
else
    LIMIT="$IDLE_MINUTES"
    REASON="nothing running"
fi

if [ "$COUNT" -ge "$LIMIT" ]; then
    logger -t idle-shutdown "GPU idle ${COUNT}m, ${REASON} — shutting down"
    wall "GPU idle ${COUNT} minutes (${REASON}). Shutting down. Prevent with: touch ${HOLD}" 2>/dev/null
    sleep 10
    /sbin/shutdown -h now
fi
