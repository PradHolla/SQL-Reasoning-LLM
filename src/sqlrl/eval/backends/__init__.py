"""Pluggable generation backends.

Two exist on purpose. ``hf`` is transformers + peft: the reference
implementation, simple enough to trust and slow enough to hurt. ``vllm`` is the
one we can afford to run repeatedly.

They must agree before vLLM numbers are trusted -- sampling and stop-token
handling differ subtly between the two, and that difference shows up as an
accuracy delta that looks like a real result.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sqlrl.eval.prompts import Prompt

__all__ = ["Backend"]


@runtime_checkable
class Backend(Protocol):
    """Turns prompts into raw generated text. Nothing else."""

    #: Label used in reports.
    name: str

    def generate(self, prompts: list[Prompt], max_new_tokens: int = 512) -> list[str]:
        """Generate one completion per prompt, in the same order."""
        ...
