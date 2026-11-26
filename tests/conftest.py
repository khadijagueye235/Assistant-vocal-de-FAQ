"""Éléments partagés par tous les tests."""

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Audio synthétique (Edge TTS, voix fr-FR-DeniseNeural) de 4,8 s.
AUDIO_REFERENCE = FIXTURES / "question_retrait.mp3"
TEXTE_REFERENCE = "Bonjour, comment puis-je retirer mon argent envoyé par Western Union ?"

besoin_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg n'est pas installé"
)


@pytest.fixture
def audio_reference() -> Path:
    return AUDIO_REFERENCE
