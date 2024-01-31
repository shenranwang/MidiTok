"""REMI (Revamped MIDI) tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from symusic import (
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

from miditok.tokenizations.remi import REMI

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
    ) -> None:
        super().__init__(tokenizer_config, max_bar_embedding, params)
        self.remove_from_chord_seq = ['Pitch', 'Velocity']
        # self.separate_chord_vocab()
        # self.vocab = self.instr_vocab  # WARNING! Hacky solution to adjust vocab to instr_vocab for compatibility with class functions
        # print("updated vocab:", self.vocab)
        
    def separate_chord_vocab(self):
        i, j = 0, 0
        self.instr_vocab, self.chord_vocab = {}, {}
        for k,v in self.vocab.items():
            if self.token_id_type(v) != 'Chord':
                self.instr_vocab[k] = i
                i += 1
            if self.token_id_type(v) not in self.remove_from_chord_seq:
                self.chord_vocab[k] = j
                j += 1
                
    # @property
    # def vocab(self) -> dict[str, int] | list[dict[str, int]]:
    #     return self._vocab_base
                
    # @vocab.setter
    # def vocab(self, value):
    #     """Setter for the vocab property."""
    #     self._vocab_base = value

    def _create_token_types_graph(self) -> dict[str, list[str]]:
        dic = super()._create_token_types_graph()
        dic["Chord"] += ["Duration"]
        return dic