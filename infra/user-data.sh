#!/usr/bin/env bash
# Instance bootstrap. Runs once, as root, on first boot.
#
# Placeholders (__REGION__, __BUCKET__, __REPO_URL__, ...) are substituted by
# launch.sh before this is handed to EC2.
#
# Everything is logged to /var/log/user-data.log — user-data failures are silent
# otherwise, and debugging a half-booted box without that log is miserable.

exec > >(tee -a /var/log/user-data.log) 2>&1
set -x

REGION="__REGION__"
BUCKET="__BUCKET__"
REPO_URL="__REPO_URL__"
IDLE_SHUTDOWN_MINUTES="__IDLE_SHUTDOWN_MINUTES__"
IDLE_GPU_PCT="__IDLE_GPU_PCT__"

APP_USER=ubuntu
APP_DIR=/opt/sql-llm
REPO_DIR="$APP_DIR/repo"
HF_CACHE=/opt/hf-cache

# --- packages -------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git jq unzip curl

# The base driver AMI does not ship the AWS CLI.
if ! command -v aws >/dev/null 2>&1; then
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install
fi
export PATH=/usr/local/bin:$PATH

# --- layout ---------------------------------------------------------------
mkdir -p "$APP_DIR" "$HF_CACHE"
chown -R "$APP_USER:$APP_USER" "$APP_DIR" "$HF_CACHE"

# --- repo -----------------------------------------------------------------
sudo -u "$APP_USER" git clone "$REPO_URL" "$REPO_DIR"

# --- secrets from SSM Parameter Store -------------------------------------
# Fetched via the instance role, never baked into user-data (user-data is
# readable by anything that can reach the metadata service).
ENV_FILE="$REPO_DIR/.env"
: > "$ENV_FILE"

HF=$(aws ssm get-parameter --region "$REGION" --name /sql-llm/HF_TOKEN \
     --with-decryption --query Parameter.Value --output text 2>/dev/null || true)
if [ -n "$HF" ] && [ "$HF" != "None" ]; then
    echo "HF_TOKEN=$HF" >> "$ENV_FILE"
fi

WB=$(aws ssm get-parameter --region "$REGION" --name /sql-llm/WANDB_API_KEY \
     --with-decryption --query Parameter.Value --output text 2>/dev/null || true)
if [ -n "$WB" ] && [ "$WB" != "None" ]; then
    echo "WANDB_API_KEY=$WB" >> "$ENV_FILE"
fi

chown "$APP_USER:$APP_USER" "$ENV_FILE"
chmod 600 "$ENV_FILE"

# --- shell environment ----------------------------------------------------
cat > /etc/profile.d/sql-llm.sh <<EOF
export HF_HOME=$HF_CACHE
export SQL_LLM_BUCKET=$BUCKET
export SQL_LLM_DIR=$REPO_DIR
export AWS_DEFAULT_REGION=$REGION
export PATH=\$HOME/.local/bin:/usr/local/bin:\$PATH
cd $REPO_DIR 2>/dev/null || true
EOF
chmod 644 /etc/profile.d/sql-llm.sh

# --- uv + python deps -----------------------------------------------------
sudo -u "$APP_USER" bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -u "$APP_USER" bash -lc "cd $REPO_DIR && \$HOME/.local/bin/uv sync" || \
    echo "WARNING: uv sync failed — resolve manually after SSH"

# --- guardrail: idle auto-shutdown ---------------------------------------
install -m 755 "$REPO_DIR/infra/idle-shutdown.sh" /usr/local/bin/idle-shutdown.sh
cat > /etc/cron.d/sql-llm-idle <<EOF
IDLE_SHUTDOWN_MINUTES=$IDLE_SHUTDOWN_MINUTES
IDLE_GPU_PCT=$IDLE_GPU_PCT
* * * * * root /usr/local/bin/idle-shutdown.sh
EOF
chmod 644 /etc/cron.d/sql-llm-idle

# --- guardrail: spot interruption rescue ----------------------------------
install -m 755 "$REPO_DIR/infra/spot-watch.sh" /usr/local/bin/spot-watch.sh
cat > /etc/systemd/system/spot-watch.service <<EOF
[Unit]
Description=Rescue checkpoints to S3 on spot interruption
After=network-online.target

[Service]
Type=simple
Environment=SQL_LLM_BUCKET=$BUCKET
Environment=AWS_DEFAULT_REGION=$REGION
Environment=WATCH_DIR=$REPO_DIR/outputs
ExecStart=/usr/local/bin/spot-watch.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now spot-watch.service

# --- done -----------------------------------------------------------------
touch "$APP_DIR/.bootstrap-complete"
chown "$APP_USER:$APP_USER" "$APP_DIR/.bootstrap-complete"
echo "BOOTSTRAP COMPLETE at $(date -u)"
