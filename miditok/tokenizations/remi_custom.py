"""REMI (Revamped MIDI) tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from symusic import Score

from miditok.tokenizations.remi import REMI
from miditok.classes import TokenizerConfig, TokSequence

if TYPE_CHECKING:
    from pathlib import Path


class REMICustom(REMI):
    r"""
    REMI tokenizer, with token types graph adjusted for use case where chords are separated entirely into their own sequence.
    """
    
    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        max_bar_embedding: int | None = None,
        params: str | Path | None = None,
        remove_token_types_from_chord_seq: list[str] = ['Pitch', 'Velocity', 'Duration'],
        remove_token_types_from_instr_seq: list[str] = ['Chord'],
    ) -> None:
        super().__init__(tokenizer_config, max_bar_embedding, params)
        self.remove_from_chord_seq = remove_token_types_from_chord_seq
        self.remove_from_instr_seq = remove_token_types_from_instr_seq
        self.instr_vocab, self.chord_vocab = {}, {}
        self.separate_chord_vocab()
        # print("Instrument vocab:", self.instr_vocab)
        # print("Chord vocab:", self.chord_vocab)

    def separate_chord_vocab(self):
        i, j = 0, 0
        for k,v in self.vocab.items():
            if self.token_id_type(v) not in self.remove_from_instr_seq:
                self.instr_vocab[k] = i
                i += 1
            if self.token_id_type(v) not in self.remove_from_chord_seq:
                self.chord_vocab[k] = j
                j += 1
                
    @property
    def vocab(self) -> dict[str, int] | list[dict[str, int]]:
        return self._vocab_base
                
    @vocab.setter
    def vocab(self, value):
        """Setter for the vocab property."""
        self._vocab_base = value

    def get_disabled_indices(self, token_types) -> list[int]:
        return sum([self.token_ids_of_type(t) for t in token_types], [])

    @property
    def get_disabled_instr_indices(self) -> list[int]:
        return self.get_disabled_indices(self.remove_from_instr_seq)

    @property
    def get_disabled_chord_indices(self) -> list[int]:
        return self.get_disabled_indices(self.remove_from_chord_seq)

    def remove_extra_tokens_from_chord_seq(self, seq: list) -> list:
        return remove_token_types_from_seq(self, seq, self.remove_from_chord_seq)

    def remove_extra_tokens_from_instr_seq(self, seq: list) -> list:
        return remove_token_types_from_seq(self, seq, self.remove_from_instr_seq)

    def func_to_get_labels(self, score: Score, tseq: TokSequence | list[TokSequence], file_path: Path) -> list[int]:
        """Return the other track."""
        return tseq[1].ids

    def _score_to_tokens(self, score: Score) -> TokSequence | list[TokSequence]:
        tok_sequence = super()._score_to_tokens(score)
        # tok_sequence[0].ids = [self.instr_vocab[self.vocab[t]] for t in self.remove_extra_tokens_from_instr_seq(instr_seq)]
        # tok_sequence[1].ids = [self.chord_vocab[self.vocab[t]] for t in self.remove_extra_tokens_from_chord_seq(chord_seq)]
        instr_seq = tok_sequence[0].ids
        tok_sequence[0].ids = self.remove_extra_tokens_from_instr_seq(instr_seq)
        if len(tok_sequence) == 2: # chord seq available when training
            chord_seq = tok_sequence[1].ids
            tok_sequence[1].ids = self.remove_extra_tokens_from_chord_seq(chord_seq)
        return tok_sequence


def remove_token_types_from_seq(tokenizer, sequence: list, token_types_to_remove: list[str]) -> list:
    seq = [t for t in sequence if tokenizer.token_id_type(t) not in token_types_to_remove]
    return remove_consecutive_duplicates(seq)


def remove_consecutive_duplicates(lst: list) -> list:
    result = [lst[0]] if lst else []
    for item in lst[1:]:
        if item != result[-1]:
            result.append(item)
    return result
