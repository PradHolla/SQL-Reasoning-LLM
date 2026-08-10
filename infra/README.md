# Infrastructure

One-command GPU boxes for training and evaluation, with cost guardrails.

## Daily use

```bash
./infra/launch.sh              # on-demand g5.xlarge, ~$1.006/hr
./infra/launch.sh --spot       # spot, ~$0.36/hr (needs the spot quota)
./infra/status.sh              # what's running, spend, quota, spot prices
./infra/stop.sh                # stop everything (keeps the disk)
```

`launch.sh` prints the SSH command as soon as the instance is up. Bootstrap
continues for a few minutes after that; the box is ready when
`/opt/sql-llm/.bootstrap-complete` exists.

```bash
ssh -i ~/.ssh/sql-llm-ue2.pem ubuntu@<ip> 'tail -f /var/log/user-data.log'
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

Created once, in **us-east-2** (cheapest `g5` spot at setup time — roughly 30%
under us-east-1).

| Resource | Name / ID |
|---|---|
| Key pair | `sql-llm-ue2` (ed25519) → `~/.ssh/sql-llm-ue2.pem` |
| Security group | `sg-0c9d9c853ea1d03b7` — SSH from one IP only |
| IAM role | `sql-llm-ec2-role` (scoped S3 + SSM read + SSM Session Manager) |
| Instance profile | `sql-llm-ec2-profile` |
| S3 bucket | `sql-reasoning-llm-381492233412-us-east-2` |
| Budget | `sql-reasoning-llm-ec2`, $100/month |
| VPC | `vpc-050a6be2968570ada` (default) |

All of it is centralized in `config.sh`.

## Secrets

Secrets live in **SSM Parameter Store** as `SecureString`, never in user-data
(which is readable by anything that can reach the metadata service) and never in
git. The instance role grants read access to `/sql-llm/*`, and `user-data.sh`
materializes them into `.env` at boot.

```bash
# already stored
/sql-llm/HF_TOKEN

# add W&B before the first training run
aws ssm put-parameter --region us-east-2 --name /sql-llm/WANDB_API_KEY \
    --value "<key>" --type SecureString --overwrite
```

## Gotchas

**Your IP changes.** The security group allows SSH from one address. When your
network changes, SSH hangs. Fix:

```bash
aws ec2 authorize-security-group-ingress --region us-east-2 \
    --group-id sg-0c9d9c853ea1d03b7 --protocol tcp --port 22 \
    --cidr "$(curl -s https://checkip.amazonaws.com)/32"
```

The instance role also includes SSM Session Manager, so
`aws ssm start-session --target <instance-id>` works with no open ports at all.

**Spot vs on-demand shutdown behaviour.** On-demand instances *stop* on
shutdown, so the root volume and your work survive. Spot instances *terminate* —
anything not in S3 is gone. Treat a spot box as disposable.

**Stopped instances still bill EBS**, about $16/month for the 200 GB gp3 root
volume. Terminate boxes you are done with.
