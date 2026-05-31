"""Kokoro — pluggable emotional trajectory memory layer for AI companions."""

__version__ = "0.1.0"

from kokoro.memory import WorldMemory
from kokoro.store import StateStore
from kokoro.decoder import StateDecoder
from kokoro.retrieval import MemoryStore
from kokoro.encoder import SessionEncoder
from kokoro.transition import TransitionModel

__all__ = [
    "WorldMemory",
    "StateStore",
    "StateDecoder",
    "MemoryStore",
    "SessionEncoder",
    "TransitionModel",
]
