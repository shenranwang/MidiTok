"""
Root module.

Here we only import tokenizer classes and submodules.
"""

from miditok import data_augmentation

from .classes import Event, TokenizerConfig, TokSequence
from .midi_tokenizer import MIDITokenizer
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
from .utils import utils

__all__ = [
    "MIDITokenizer",
    "Event",
    "TokSequence",
    "TokenizerConfig",
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
