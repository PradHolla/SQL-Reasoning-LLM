# Infrastructure

One-command GPU boxes for training and evaluation, with cost guardrails.

## Daily use

```bash
./infra/launch.sh              # new on-demand g5.xlarge, ~$1.006/hr
./infra/launch.sh --spot       # spot, ~$0.48/hr (needs the spot quota)
./infra/start.sh               # restart a stopped box (keeps repo, venv, HF cache)
./infra/status.sh              # what's running, spend, quota, spot prices
./infra/stop.sh                # stop everything (keeps the disk)
./infra/stop.sh --terminate    # destroy it, disk included
```

After an idle shutdown use `start.sh`, not `launch.sh`: a stopped instance keeps
its root volume, so the repo, the uv environment and the ~2 GB HuggingFace cache
survive. Relaunching downloads all of it again. The public IP changes on every
start, so `start.sh` prints the new one and refreshes the SSH rule.

`launch.sh` prints the SSH command as soon as the instance is up. Bootstrap
continues for a few minutes after that; the box is ready when
`/opt/sql-llm/.bootstrap-complete` exists.

```bash
ssh -i ~/.ssh/sql-llm-ue1.pem ubuntu@<ip> 'tail -f /var/log/user-data.log'
```

## What the guardrails do

**Idle auto-shutdown** — a cron job checks GPU utilization every minute and
shuts the instance down after 30 minutes below 5%. A forgotten `g5.xlarge` is
about $170/week; this caps that. Override during long CPU-only work:

```bash
touch /opt/sql-llm/.no-autoshutdown
```

**Spot interruption rescue** — a systemd service polls the metadata service and,
on the two-minute interruption warning, syncs `outputs/` to S3 before the
instance disappears. It is not a substitute for `save_steps` + S3 sync in the
trainer itself; it is the last line of defence.

**Budget** — account-wide $100/month with alerts at 50%, 80%, 100% actual and
100% forecast, to `prad@pradholla.com`.

## Provisioned resources

Everything lives in **us-east-1**.

| Resource | Name / ID |
|---|---|
| Key pair | `sql-llm` (rsa) → `~/.ssh/sql-llm-ue1.pem` |
| Security group | `sg-02ff9b697e95494e6` — SSH from one IP only |
| IAM role | `sql-llm-ec2-role` (scoped S3 + SSM read + SSM Session Manager) |
| Instance profile | `sql-llm-ec2-profile` (IAM is global) |
| S3 bucket | `sql-reasoning-llm-381492233412-us-east-1` |
| Budget | `sql-reasoning-llm-ec2`, $100/month (account-wide) |
| VPC | `vpc-045cc999e6dbf6bf2` (default) |

All of it is centralized in `config.sh`.

### Why not us-east-2

The project was originally provisioned in us-east-2, because `g5` spot ran
about 30% cheaper there. That turned out not to matter: **us-east-2 has zero
G/VT quota of either kind**, on-demand included, so it cannot launch a GPU box
at all. us-east-1 has 8 vCPU of on-demand G/VT approved.

Running compute in us-east-1 against a us-east-2 bucket and secrets was
considered and rejected. It would save roughly $0.13/hr on spot pricing we have
no quota for, in exchange for a permanent "which region is that in?" tax on
every future debugging session. Quota beats list price, and one region beats
two.

## Secrets

Secrets live in **SSM Parameter Store** as `SecureString`, never in user-data
(which is readable by anything that can reach the metadata service) and never in
git. The instance role grants read access to `/sql-llm/*`, and `user-data.sh`
materializes them into `.env` at boot.

```bash
# stored, in us-east-1
/sql-llm/HF_TOKEN
/sql-llm/WANDB_API_KEY

# to rotate one
aws ssm put-parameter --region us-east-1 --name /sql-llm/WANDB_API_KEY \
    --value "<key>" --type SecureString --overwrite
```

## Gotchas

**Your IP changes.** The security group allows SSH from one address. When your
network changes, SSH hangs. Fix:

```bash
aws ec2 authorize-security-group-ingress --region us-east-1 \
    --group-id sg-02ff9b697e95494e6 --protocol tcp --port 22 \
    --cidr "$(curl -s https://checkip.amazonaws.com)/32"
```

The instance role also includes SSM Session Manager, so
`aws ssm start-session --target <instance-id>` works with no open ports at all.

**Spot vs on-demand shutdown behaviour.** On-demand instances *stop* on
shutdown, so the root volume and your work survive. Spot instances *terminate* —
anything not in S3 is gone. Treat a spot box as disposable.

**Stopped instances still bill EBS**, about $16/month for the 200 GB gp3 root
volume. Terminate boxes you are done with.
