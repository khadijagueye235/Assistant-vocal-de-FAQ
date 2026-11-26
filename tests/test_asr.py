"""ASR : préparation de l'audio (rapide) et transcription (charge Whisper)."""

import subprocess

import numpy as np
import pytest
from conftest import TEXTE_REFERENCE, besoin_ffmpeg

from src import asr


def _stereo_int16(chemin, frequence: int) -> np.ndarray:
    """Décode un fichier comme le ferait Gradio : int16, stéréo."""
    brut = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(chemin),
            "-f",
            "s16le",
            "-ac",
            "2",
            "-ar",
            str(frequence),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(brut, dtype=np.int16).reshape(-1, 2)


# --- Préparation de l'audio (sans modèle) ---


@besoin_ffmpeg
def test_fichier_decode_en_mono_16k(audio_reference):
    signal = asr.preparer_audio(audio_reference)
    assert signal.dtype == np.float32 and signal.ndim == 1
    assert len(signal) / asr.FREQUENCE_HZ == pytest.approx(4.78, abs=0.05)


@besoin_ffmpeg
def test_tuple_gradio_stereo_48k_reechantillonne(audio_reference):
    stereo = _stereo_int16(audio_reference, 48_000)
    signal = asr.preparer_audio((48_000, stereo))
    assert signal.dtype == np.float32 and signal.ndim == 1
    assert len(signal) == pytest.approx(len(stereo) / 3, rel=0.01)  # 48 kHz -> 16 kHz
    assert np.abs(signal).max() <= 1.0


def test_tuple_16k_converti_sans_reechantillonnage():
    entiers = np.array([0, 16384, -32768, 32767], dtype=np.int16)
    signal = asr.preparer_audio((16_000, entiers))
    np.testing.assert_allclose(signal, [0.0, 0.5, -1.0, 32767 / 32768])


def test_tableau_numpy_passe_tel_quel():
    signal = np.linspace(-1, 1, 1000, dtype=np.float32)
    assert asr.preparer_audio(signal) is signal


@pytest.mark.parametrize(
    "entree, erreur",
    [
        ((48_000, np.array([], dtype=np.int16)), ValueError),
        (np.array([], dtype=np.float32), ValueError),
        ("fichier_absent.wav", FileNotFoundError),
    ],
)
def test_entrees_invalides(entree, erreur):
    with pytest.raises(erreur):
        asr.preparer_audio(entree)


def test_message_clair_sans_ffmpeg(monkeypatch, audio_reference):
    monkeypatch.setattr(asr.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg est introuvable"):
        asr.preparer_audio(audio_reference)


# --- Transcription (charge Whisper) ---


@pytest.mark.modele
@besoin_ffmpeg
def test_transcription_reference(audio_reference):
    resultat = asr.transcrire(audio_reference)
    assert resultat.texte == TEXTE_REFERENCE
    assert resultat.temps_s > 0


@pytest.mark.modele
def test_silence_donne_transcription_vide():
    resultat = asr.transcrire(np.zeros(2 * asr.FREQUENCE_HZ, dtype=np.float32))
    assert resultat.est_vide


def test_avertissement_visible_sur_cpu(monkeypatch, caplog):
    """Un basculement silencieux vers le CPU fausserait les latences mesurées."""
    monkeypatch.setattr(asr, "_choisir_materiel", lambda: ("cpu", "int8"))
    monkeypatch.setattr(asr, "WhisperModel", lambda *args, **kwargs: "faux modèle")
    asr.charger_modele.cache_clear()
    try:
        asr.charger_modele("tiny")
    finally:
        asr.charger_modele.cache_clear()  # ne pas laisser le faux modèle en cache
    assert "CPU" in caplog.text
    assert caplog.records[0].levelname == "WARNING"
