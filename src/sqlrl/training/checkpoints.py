"""Make a training run survive losing its machine.

Spot instances are 50-60% cheaper and can be reclaimed with a **two-minute
warning**, and reclamation means *terminate*, not stop: the root volume goes
with it. So "resume" here is not "restart the same box". It is:

    new box  ->  pull the last checkpoint from S3  ->  carry on from that step

`infra/spot-watch.sh` already rescues `outputs/` when the warning arrives. That
covers the graceful case and nothing else -- a missed notice, a hard failure, or
a kernel panic loses everything since the run began. So checkpoints also go to
S3 **on every save**, which turns the worst case from "lose the run" into "lose
the steps since the last save".

Three details that are easy to get wrong:

* **Never sync with ``--delete``.** The obvious way to stop S3 accumulating
  checkpoints is to mirror the local directory, which respects
  ``save_total_limit`` for free. It is also one fresh instance away from
  deleting the only copy of a run, because a new box's checkpoint directory is
  empty and an empty mirror deletes everything. Old checkpoints are ~110 MB of
  LoRA adapter plus optimiser state; letting them pile up is much cheaper than
  the failure mode.
* **The bucket is a constant, not an environment variable.** This project has
  lost work three separate times to variables that were set in a login shell and
  invisible to a job, including one S3 sync that silently uploaded nothing
  because ``$SQL_LLM_BUCKET`` was unset and ``--quiet`` hid it. The name is in
  ``infra/config.sh`` in this repo already; it is not a secret.
* **W&B needs a fixed run id.** Without one, a resumed run starts a second chart
  and the loss curve arrives in fragments with a gap in the middle.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from transformers import TrainerCallback

__all__ = [
    "BUCKET",
    "S3CheckpointSync",
    "latest_checkpoint",
    "pull_checkpoints",
    "push_checkpoints",
    "resume_from",
    "wandb_resumable",
]

#: Same bucket as infra/config.sh. Overridable, but with a working default, so a
#: job started in a non-login shell still syncs somewhere real.
BUCKET = os.environ.get("SQL_LLM_BUCKET", "sql-reasoning-llm-381492233412-us-east-1")

_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(directory: Path) -> Path | None:
    """The highest-numbered ``checkpoint-N`` under ``directory``, if any."""
    if not directory.is_dir():
        return None
    numbered = [
        (int(match.group(1)), path)
        for path in directory.iterdir()
        if path.is_dir() and (match := _CHECKPOINT.match(path.name))
    ]
    return max(numbered)[1] if numbered else None


def _s3_uri(run_name: str, bucket: str = BUCKET) -> str:
    return f"s3://{bucket}/checkpoints/{run_name}"


def push_checkpoints(local: Path, run_name: str, bucket: str = BUCKET) -> None:
    """Mirror the checkpoint directory up to S3. Never deletes -- see module docs."""
    if not local.is_dir():
        return
    subprocess.run(
        ["aws", "s3", "sync", str(local), _s3_uri(run_name, bucket)],
        check=False,  # a failed sync must not kill a training run
    )


def pull_checkpoints(local: Path, run_name: str, bucket: str = BUCKET) -> Path | None:
    """Fetch checkpoints for ``run_name`` from S3, returning the latest locally."""
    local.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["aws", "s3", "sync", _s3_uri(run_name, bucket), str(local)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  S3 pull failed: {result.stderr.strip()[:200]}")
        return None
    return latest_checkpoint(local)


class S3CheckpointSync(TrainerCallback):
    """Push checkpoints to S3 every time the trainer writes one.

    `on_save` fires after the checkpoint is complete on disk, so there is no
    window where a half-written directory is uploaded.
    """

    def __init__(self, local: Path, run_name: str, bucket: str = BUCKET) -> None:
        self.local = Path(local)
        self.run_name = run_name
        self.bucket = bucket

    def on_save(self, args, state, control, **kwargs) -> None:
        print(f"  syncing checkpoints -> {_s3_uri(self.run_name, self.bucket)}",
              flush=True)
        push_checkpoints(self.local, self.run_name, self.bucket)


def resume_from(
    checkpoints: Path, run_name: str, resume: bool, bucket: str = BUCKET
) -> str | bool | None:
    """What to hand `trainer.train(resume_from_checkpoint=...)`.

    Returns ``None`` for a fresh run, or a checkpoint path to continue from.
    Pulls from S3 first, because on spot the machine that wrote the checkpoint
    is usually gone.

    Deliberately loud when ``--resume`` is asked for and nothing is found:
    silently starting from scratch would waste the whole run and look, from the
    logs, exactly like a resume that worked.
    """
    if not resume:
        return None

    found = latest_checkpoint(checkpoints)
    if found is None:
        print(f"no local checkpoint in {checkpoints}; pulling from S3...")
        found = pull_checkpoints(checkpoints, run_name, bucket)

    if found is None:
        raise FileNotFoundError(
            f"--resume was requested but no checkpoint exists for run "
            f"{run_name!r}, locally in {checkpoints} or at "
            f"{_s3_uri(run_name, bucket)}. Refusing to silently start from "
            f"scratch: that wastes the run and reads like a successful resume."
        )
    print(f"resuming from {found}")
    return str(found)


def wandb_resumable(run_name: str) -> None:
    """Pin the W&B run id so a resumed run continues one chart, not two.

    Set before the trainer builds its integrations. Without it, an interrupted
    run's curves arrive as two separate runs with a gap, which is exactly when
    you most want to read them as one.
    """
    os.environ.setdefault("WANDB_RUN_ID", re.sub(r"[^A-Za-z0-9_.-]", "-", run_name))
    os.environ.setdefault("WANDB_RESUME", "allow")
