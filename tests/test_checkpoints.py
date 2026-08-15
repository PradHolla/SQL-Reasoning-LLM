"""Tests for sqlrl.training.checkpoints.

This is spot-instance resume machinery: a reclaimed spot instance is
*terminated*, so the only surviving copy of a checkpoint is whatever made it
to S3. Every test here pins one way this could go silently wrong -- a
lexicographic sort that resumes from step 9 instead of step 100, a
``--delete`` sync that wipes the only copy of a run, or a resume that finds
nothing and quietly starts over instead of raising. All of those look, from
the training logs, exactly like a run that is working correctly, which is
what makes them worth a test each.

No test here makes a real network or AWS call: ``subprocess.run`` is
monkeypatched throughout, so nothing needs AWS credentials to pass.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from sqlrl.training import checkpoints
from sqlrl.training.checkpoints import (
    BUCKET,
    S3CheckpointSync,
    latest_checkpoint,
    push_checkpoints,
    resume_from,
    wandb_resumable,
)


class RecordingRun:
    """Stand-in for ``subprocess.run`` that records commands instead of
    running them, so push/pull tests can never reach the network or need AWS
    credentials.
    """

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr=self.stderr)


# --------------------------------------------------------------------------
# latest_checkpoint
# --------------------------------------------------------------------------


def test_latest_checkpoint_compares_numerically_not_lexicographically(tmp_path):
    # String sorting would put "checkpoint-9" after "checkpoint-100" and
    # "checkpoint-50", silently resuming a run from thousands of steps
    # earlier than the actual latest save.
    for n in (9, 50, 100):
        (tmp_path / f"checkpoint-{n}").mkdir()
    assert latest_checkpoint(tmp_path).name == "checkpoint-100"


def test_latest_checkpoint_returns_none_for_a_missing_directory(tmp_path):
    assert latest_checkpoint(tmp_path / "does-not-exist") is None


def test_latest_checkpoint_returns_none_for_an_empty_directory(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_returns_none_for_only_non_checkpoint_entries(tmp_path):
    (tmp_path / "README.md").write_text("hi")
    (tmp_path / "checkpoint-abc").mkdir()  # doesn't match \d+, not a checkpoint
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_ignores_files_named_like_checkpoints(tmp_path):
    # Only directories count. A file that happens to be named like the
    # highest-numbered checkpoint must not win over a real one.
    (tmp_path / "checkpoint-50").mkdir()
    (tmp_path / "checkpoint-100").write_text("not a checkpoint directory")
    assert latest_checkpoint(tmp_path).name == "checkpoint-50"


# --------------------------------------------------------------------------
# push_checkpoints
# --------------------------------------------------------------------------


def test_push_checkpoints_never_passes_delete(tmp_path, monkeypatch):
    # --delete mirrors the local directory to S3, and a fresh instance's
    # local directory starts empty -- one sync away from deleting the only
    # copy of the run.
    local = tmp_path / "checkpoints"
    local.mkdir()
    fake = RecordingRun()
    monkeypatch.setattr(checkpoints.subprocess, "run", fake)

    push_checkpoints(local, "my-run")

    assert len(fake.calls) == 1
    cmd = fake.calls[0]
    assert "--delete" not in cmd
    assert str(local) in cmd
    assert f"s3://{BUCKET}/checkpoints/my-run" in cmd


def test_push_checkpoints_is_a_noop_when_the_local_directory_does_not_exist(
    tmp_path, monkeypatch
):
    fake = RecordingRun()
    monkeypatch.setattr(checkpoints.subprocess, "run", fake)

    push_checkpoints(tmp_path / "does-not-exist", "my-run")

    assert fake.calls == []


def test_push_checkpoints_does_not_raise_when_the_sync_fails(tmp_path, monkeypatch):
    # A failed upload must never kill an otherwise-healthy training run.
    local = tmp_path / "checkpoints"
    local.mkdir()
    fake = RecordingRun(returncode=1, stderr="network unreachable")
    monkeypatch.setattr(checkpoints.subprocess, "run", fake)

    push_checkpoints(local, "my-run")  # must not raise


# --------------------------------------------------------------------------
# S3CheckpointSync
# --------------------------------------------------------------------------


def test_s3_checkpoint_sync_on_save_pushes_its_configured_directory(tmp_path, monkeypatch):
    local = tmp_path / "checkpoints"
    local.mkdir()
    fake = RecordingRun()
    monkeypatch.setattr(checkpoints.subprocess, "run", fake)

    sync = S3CheckpointSync(local, "my-run")
    sync.on_save(None, None, None)

    assert len(fake.calls) == 1
    cmd = fake.calls[0]
    assert str(local) in cmd
    assert f"s3://{BUCKET}/checkpoints/my-run" in cmd


# --------------------------------------------------------------------------
# resume_from
# --------------------------------------------------------------------------


def test_resume_from_with_resume_false_returns_none_without_touching_s3(tmp_path, monkeypatch):
    # A fresh run must not pay for a network round trip.
    fake = RecordingRun()
    monkeypatch.setattr(checkpoints.subprocess, "run", fake)

    result = resume_from(tmp_path / "checkpoints", "my-run", resume=False)

    assert result is None
    assert fake.calls == []


def test_resume_from_prefers_a_local_checkpoint_over_s3(tmp_path, monkeypatch):
    # The local copy is authoritative; pulling would just be wasted time.
    local = tmp_path / "checkpoints"
    (local / "checkpoint-30").mkdir(parents=True)
    fake = RecordingRun()
    monkeypatch.setattr(checkpoints.subprocess, "run", fake)

    result = resume_from(local, "my-run", resume=True)

    assert result == str(local / "checkpoint-30")
    assert fake.calls == []


def test_resume_from_pulls_from_s3_when_no_local_checkpoint_exists(tmp_path, monkeypatch):
    local = tmp_path / "checkpoints"

    def fake_run(cmd, **kwargs):
        # Simulate `aws s3 sync` populating the local directory from S3.
        (local / "checkpoint-50").mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checkpoints.subprocess, "run", fake_run)

    result = resume_from(local, "my-run", resume=True)

    assert result == str(local / "checkpoint-50")


def test_resume_from_raises_naming_run_and_s3_uri_when_the_sync_fails(tmp_path, monkeypatch):
    # The whole point: silently starting from scratch would waste a
    # multi-hour run and read, in the logs, exactly like a working resume.
    local = tmp_path / "checkpoints"
    fake = RecordingRun(returncode=1, stderr="An error occurred (403): Forbidden")
    monkeypatch.setattr(checkpoints.subprocess, "run", fake)

    with pytest.raises(FileNotFoundError) as excinfo:
        resume_from(local, "my-run", resume=True)

    message = str(excinfo.value)
    assert "my-run" in message
    assert f"s3://{BUCKET}/checkpoints/my-run" in message


def test_resume_from_raises_naming_run_and_s3_uri_when_the_bucket_is_empty(
    tmp_path, monkeypatch
):
    # Distinct failure mode from above: the sync itself can exit 0 -- there
    # was simply nothing there, e.g. a run that never got past step zero.
    # That must fail exactly the same way, not be mistaken for a fresh run.
    local = tmp_path / "checkpoints"
    fake = RecordingRun(returncode=0)
    monkeypatch.setattr(checkpoints.subprocess, "run", fake)

    with pytest.raises(FileNotFoundError) as excinfo:
        resume_from(local, "my-run", resume=True)

    message = str(excinfo.value)
    assert "my-run" in message
    assert f"s3://{BUCKET}/checkpoints/my-run" in message


# --------------------------------------------------------------------------
# wandb_resumable
# --------------------------------------------------------------------------


def test_wandb_resumable_sets_run_id_and_resume_mode(monkeypatch):
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    wandb_resumable("my-run")

    assert os.environ["WANDB_RUN_ID"] == "my-run"
    assert os.environ["WANDB_RESUME"] == "allow"


def test_wandb_resumable_sanitises_characters_wandb_ids_cannot_contain(monkeypatch):
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    wandb_resumable("qwen/grpo run 1")

    assert os.environ["WANDB_RUN_ID"] == "qwen-grpo-run-1"


def test_wandb_resumable_does_not_overwrite_an_already_set_run_id(monkeypatch):
    # A resumed run's launch script calls this again; it must not stomp on
    # a run id that a previous invocation, or the environment, already set.
    monkeypatch.setenv("WANDB_RUN_ID", "already-set")
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    wandb_resumable("my-run")

    assert os.environ["WANDB_RUN_ID"] == "already-set"
