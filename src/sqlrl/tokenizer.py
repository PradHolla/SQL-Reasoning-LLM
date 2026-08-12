"""One place that decides how text becomes tokens.

Every stage used to do this by hand, and the result only worked by accident.
The pattern in v0 was:

    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

Unsloth's `get_chat_template` overwrites `eos_token` with `<|im_end|>`, so those
first two lines did nothing at all — the correct eos arrived as a side effect of
a helper called for a different reason. Swap the order and generation silently
stops terminating: the model runs to `max_new_tokens` on every request, and the
only symptom is that evaluation numbers look bad.

That is worth removing from the project permanently, so this module sets the
template and the special tokens explicitly instead of inheriting them.

Deliberately depends on `transformers` only, never Unsloth. Unsloth is a
training-time optimization that raises NotImplementedError on non-NVIDIA/AMD/
Intel hardware, and evaluation must stay runnable on a laptop.

THE RULE: the tokenizer travels with the checkpoint
---------------------------------------------------
Always pass the directory of the weights you are about to run. Never pass a base
model name while loading an adapter trained elsewhere.

This is not general tidiness, it is specific to this project. Unsloth's
`get_chat_template` rewrote the vocabulary during v0 training and **swapped two
special token ids**:

    token            stock Qwen2.5    v0 SFT / GRPO checkpoints
    <|endoftext|>       151643              151645
    <|im_start|>        151644              151644
    <|im_end|>          151645              151643

Ordinary text is unaffected; only these specials moved. The CPT checkpoint never
had a chat template applied and kept the stock mapping.

So v0's SFT and GRPO adapters emit **151643** to end a turn. Evaluate them with a
stock tokenizer and `<|im_end|>` resolves to 151645, generation never stops, every
completion runs to max_new_tokens, and execution accuracy reads near zero for a
reason that has nothing to do with the model. Use `assert_stops_on` to catch it.
"""

from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Byte-identical to the template v0 saved to models/*/chat_template.jinja, so
# adapters trained before this module still see the prompts they were fit on.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

# Qwen2.5's natural end-of-document token; also what CPT trains against.
BASE_EOS = "<|endoftext|>"

# ChatML terminates every turn with this. It is what generation must stop on.
CHAT_EOS = "<|im_end|>"


def build_tokenizer(
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
    *,
    chat: bool = True,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer configured for exactly one of the two regimes we use.

    chat=True  — ChatML. SFT, GRPO, evaluation, serving. Turns end with
                 `<|im_end|>`, and that is the eos generation halts on.
    chat=False — raw text. Continued pretraining only, where there are no turns
                 and documents end with `<|endoftext|>`.

    Passing the wrong one is the mistake this module exists to make visible:
    a ChatML-trained model loaded with chat=False will not stop generating.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if chat:
        tokenizer.chat_template = CHATML_TEMPLATE
        tokenizer.eos_token = CHAT_EOS
        # Pad must differ from eos. When they are the same token, a padded batch
        # is indistinguishable from a batch of finished sequences, and any code
        # that masks padding also masks the real stop token.
        tokenizer.pad_token = BASE_EOS
    else:
        tokenizer.chat_template = None
        tokenizer.eos_token = BASE_EOS
        tokenizer.pad_token = BASE_EOS

    return tokenizer


def special_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """The specials and the ids they actually resolve to. Log this with every run.

    Cheap insurance: the v0 id swap would have been obvious from day one if this
    had appeared in the logs.
    """
    return {
        tok: tokenizer.convert_tokens_to_ids(tok)
        for tok in (BASE_EOS, "<|im_start|>", CHAT_EOS)
    }


def assert_stops_on(tokenizer: PreTrainedTokenizerBase, checkpoint_path: str) -> None:
    """Fail loudly if `tokenizer` disagrees with the checkpoint about the stop token.

    Call this before generating with any adapter. The failure it prevents is
    silent — the model runs to max_new_tokens and you read it as a bad score
    rather than a broken pairing.
    """
    saved = AutoTokenizer.from_pretrained(checkpoint_path)
    expected = saved.eos_token_id
    actual = tokenizer.eos_token_id

    if actual != expected:
        raise ValueError(
            f"Stop-token mismatch against {checkpoint_path!r}.\n"
            f"  checkpoint stops on: {saved.eos_token!r} (id {expected})\n"
            f"  this tokenizer uses: {tokenizer.eos_token!r} (id {actual})\n"
            f"Generation would never terminate. Build the tokenizer from the "
            f"checkpoint directory, not from a base model name."
        )
