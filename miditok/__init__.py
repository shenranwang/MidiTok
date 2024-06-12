"""
Root module.

Here we only import tokenizer classes and submodules.
"""

from miditok import data_augmentation

from .classes import Event, TokenizerConfig, TokSequence
from .midi_tokenizer import MusicTokenizer
from .tokenizations import (
    MMM,
    REMI,
    REMICustom,
    TSD,
    CPWord,
    MIDILike,
    MuMIDI,
    Octuple,
    Structured,
)
from .tokenizer_training_iterator import TokTrainingIterator

__all__ = [
    "MusicTokenizer",
    "Event",
    "TokSequence",
    "TokenizerConfig",
    "TokTrainingIterator",
    "MIDILike",
    "REMI",
    "REMICustom",
    "TSD",
    "Structured",
    "Octuple",
    "CPWord",
    "MuMIDI",
    "MMM",
    "utils",
    "data_augmentation",
]

try:
    from miditok import pytorch_data  # noqa: F401

    __all__.append("pytorch_data")
except ImportError:
    pass
