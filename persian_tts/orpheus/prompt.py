"""The Orpheus training/inference sequence format.

A training example is one flat token sequence that a causal LM can be trained on
with a plain next-token loss::

    <|begin_of_text|> <SOH> speaker: text <EOT> <EOH> <SOA> <audio codes...> <EOA> <|eot_id|>
    |<------------- prompt, loss-masked ------------->||<------ supervised ------>|

Only the audio region (plus its terminating token) is supervised. Training on
the text region too would spend the 3B model's capacity re-learning Persian text
modelling, which is not the task and competes with audio for the same weights.

The special-token ids below are Orpheus's own, inherited from the Llama-3
tokenizer's reserved region. They sit *below* :data:`AUDIO_TOKEN_BASE` so they
can never collide with an audio code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

#: Llama-3 begin-of-text. Orpheus keeps it as the sequence opener.
BOS = 128000

#: End of a turn; terminates the whole example.
EOS = 128009

#: Marks the end of the *text* payload, before the human-turn closer.
END_OF_TEXT = 128009

#: Human turn (the text prompt) delimiters.
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260

#: AI turn (the audio response) delimiters.
START_OF_AI = 128261
END_OF_AI = 128262

#: Emitted right after START_OF_AI and right before END_OF_AI by the reference
#: implementation; kept so generated sequences match the pretrained prior.
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258

#: Ignored by ``CrossEntropyLoss``; marks prompt positions in ``labels``.
IGNORE_INDEX = -100

#: Padding uses Llama-3's dedicated finetune pad id, never EOS -- padding with
#: EOS teaches the model to stop early at arbitrary points.
PAD_TOKEN = 128004


@dataclass
class OrpheusExample:
    """One tokenized training example."""

    input_ids: List[int]
    labels: List[int]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                f"input_ids ({len(self.input_ids)}) and labels ({len(self.labels)}) "
                "must be the same length"
            )

    def __len__(self) -> int:
        return len(self.input_ids)


def build_prompt_ids(
    text_ids: Sequence[int], speaker: Optional[str], speaker_ids: Optional[Sequence[int]]
) -> List[int]:
    """Assemble the loss-masked prompt region.

    The speaker name is prepended *inside* the human turn as plain text
    (``"mana: <sentence>"``), which is how Orpheus conditions on voice. Using
    text rather than a learned speaker embedding means an unseen speaker name
    degrades to a nearby voice instead of indexing out of bounds, and it needs
    no architecture change to the pretrained checkpoint.
    """
    body: List[int] = []
    if speaker and speaker_ids:
        body.extend(speaker_ids)
    body.extend(text_ids)
    return [BOS, START_OF_HUMAN, *body, END_OF_TEXT, END_OF_HUMAN]


def build_example(
    text_ids: Sequence[int],
    audio_tokens: Sequence[int],
    speaker: Optional[str] = None,
    speaker_ids: Optional[Sequence[int]] = None,
) -> OrpheusExample:
    """Build a full training example with the prompt region loss-masked."""
    prompt = build_prompt_ids(text_ids, speaker, speaker_ids)
    response = [
        START_OF_AI,
        START_OF_SPEECH,
        *audio_tokens,
        END_OF_SPEECH,
        END_OF_AI,
        EOS,
    ]
    input_ids = [*prompt, *response]
    labels = [*([IGNORE_INDEX] * len(prompt)), *response]
    return OrpheusExample(input_ids=input_ids, labels=labels)


def build_generation_prompt(
    text_ids: Sequence[int],
    speaker: Optional[str] = None,
    speaker_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    """The inference-time prefix: everything up to where audio codes begin.

    Must match :func:`build_example`'s prefix exactly, or generation starts from
    a distribution the model never saw during training.
    """
    return [
        *build_prompt_ids(text_ids, speaker, speaker_ids),
        START_OF_AI,
        START_OF_SPEECH,
    ]
