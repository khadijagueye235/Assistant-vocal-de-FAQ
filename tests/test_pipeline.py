"""Contrat de repondre() : vérifié sans charger Whisper (ASR remplacé par un faux)."""

import dataclasses

import pytest

from src import pipeline
from src.asr import Transcription
from src.config import MESSAGE_RIEN_ENTENDU


@pytest.fixture
def faux_asr(monkeypatch):
    """Remplace transcrire() pour tester la logique du pipeline seule."""

    def _installer(texte: str):
        monkeypatch.setattr(
            pipeline,
            "transcrire",
            lambda audio: Transcription(texte=texte, duree_audio_s=3.0, temps_s=0.1),
        )

    return _installer


def test_question_comprise(faux_asr):
    faux_asr("comment retirer mon argent")
    r = pipeline.repondre("peu_importe.wav")
    assert r.transcription == "comment retirer mon argent"
    assert r.simulation is True
    assert r.audio_reponse.is_file()
    assert r.reponse_texte


def test_rien_entendu_ne_lance_pas_la_recherche(faux_asr):
    faux_asr("")
    r = pipeline.repondre("peu_importe.wav")
    assert r.reponse_texte == MESSAGE_RIEN_ENTENDU
    assert not r.a_trouve
    assert r.score is None
    assert r.latences.recherche_s == 0.0
    assert r.audio_reponse.is_file()  # le message est quand même dit à voix haute


def test_latences_coherentes(faux_asr):
    faux_asr("une question")
    lat = pipeline.repondre("peu_importe.wav").latences
    assert min(lat.asr_s, lat.recherche_s, lat.tts_s) >= 0
    assert lat.total_s >= max(lat.asr_s, lat.recherche_s, lat.tts_s)


def test_vers_ligne_a_plat(faux_asr):
    faux_asr("une question")
    ligne = pipeline.repondre("peu_importe.wav").vers_ligne()
    assert {"transcription", "id_question", "score", "asr_s", "total_s"} <= set(ligne)
    assert isinstance(ligne["audio_reponse"], str)
    assert not any(isinstance(v, dict) for v in ligne.values())


def test_reponse_non_modifiable(faux_asr):
    faux_asr("une question")
    r = pipeline.repondre("peu_importe.wav")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.transcription = "modifiée"
