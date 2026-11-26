"""Contrat de repondre() : vérifié sans modèle ni réseau (ASR, recherche et TTS remplacés)."""

import dataclasses

import pytest

from src import pipeline
from src.asr import Transcription
from src.config import MESSAGE_JE_NE_SAIS_PAS, MESSAGE_RIEN_ENTENDU, SEUIL_SIMILARITE
from src.search import Resultat, charger_faq
from src.tts import ErreurTTS

FAQ = {e.id: e for e in charger_faq()}


@pytest.fixture
def textes_dits(monkeypatch, tmp_path):
    """Remplace parler() : écrit un faux fichier et note chaque texte lu à voix haute."""
    dits = []

    def faux_parler(texte):
        dits.append(texte)
        chemin = tmp_path / f"reponse_{len(dits)}.mp3"
        chemin.write_bytes(b"faux mp3")
        return chemin

    monkeypatch.setattr(pipeline, "parler", faux_parler)
    return dits


@pytest.fixture
def faux_pipeline(monkeypatch, textes_dits):
    """Remplace transcrire(), chercher() et parler() pour tester la logique du pipeline seule.

    Usage : faux_pipeline("texte transcrit", id_trouve=23, score=0.95)
    """
    appels_recherche = []

    def _installer(texte: str, id_trouve: int = 23, score: float = 0.95):
        monkeypatch.setattr(
            pipeline,
            "transcrire",
            lambda audio: Transcription(texte=texte, duree_audio_s=3.0, temps_s=0.1),
        )

        def faux_chercher(requete):
            appels_recherche.append(requete)
            return Resultat(FAQ[id_trouve], score)

        monkeypatch.setattr(pipeline, "chercher", faux_chercher)
        return appels_recherche

    return _installer


def test_question_retrouvee_au_dessus_du_seuil(faux_pipeline):
    faux_pipeline("comment retirer mon argent", id_trouve=23, score=SEUIL_SIMILARITE + 0.05)
    r = pipeline.repondre("peu_importe.wav")
    assert r.a_trouve
    assert r.id_question == 23
    assert r.question_trouvee == FAQ[23].question
    assert r.reponse_texte == FAQ[23].reponse  # version écrite affichée
    assert r.score == pytest.approx(SEUIL_SIMILARITE + 0.05)
    assert r.simulation is False  # plus aucune étape simulée
    assert r.audio_reponse.is_file()


def test_la_voix_lit_la_version_orale(faux_pipeline, textes_dits):
    faux_pipeline("une question", id_trouve=23)
    pipeline.repondre("peu_importe.wav")
    assert textes_dits == [FAQ[23].reponse_vocale]  # sigles épelés, pas la version écrite


def test_je_ne_sais_pas_est_dit_a_voix_haute(faux_pipeline, textes_dits):
    faux_pipeline("quelle heure est-il", score=SEUIL_SIMILARITE - 0.01)
    pipeline.repondre("peu_importe.wav")
    assert textes_dits == [MESSAGE_JE_NE_SAIS_PAS]


def test_panne_tts_garde_la_reponse_ecrite(faux_pipeline, monkeypatch, caplog):
    faux_pipeline("comment retirer mon argent", id_trouve=23)

    def tts_en_panne(texte):
        raise ErreurTTS("réseau coupé")

    monkeypatch.setattr(pipeline, "parler", tts_en_panne)
    r = pipeline.repondre("peu_importe.wav")
    assert r.audio_reponse is None
    assert r.reponse_texte == FAQ[23].reponse
    assert r.vers_ligne()["audio_reponse"] == ""
    assert "sans voix" in caplog.text  # la panne laisse une trace dans le journal


def test_je_ne_sais_pas_sous_le_seuil(faux_pipeline):
    faux_pipeline("quelle heure est-il", id_trouve=28, score=SEUIL_SIMILARITE - 0.01)
    r = pipeline.repondre("peu_importe.wav")
    assert not r.a_trouve
    assert r.question_trouvee is None
    assert r.reponse_texte == MESSAGE_JE_NE_SAIS_PAS
    assert r.score == pytest.approx(SEUIL_SIMILARITE - 0.01)  # gardé pour l'analyse


def test_seuil_atteint_exactement_donne_une_reponse(faux_pipeline):
    faux_pipeline("une question", id_trouve=5, score=SEUIL_SIMILARITE)
    assert pipeline.repondre("peu_importe.wav").id_question == 5


def test_rien_entendu_ne_lance_pas_la_recherche(faux_pipeline, textes_dits):
    appels = faux_pipeline("")
    r = pipeline.repondre("peu_importe.wav")
    assert appels == []
    assert r.reponse_texte == MESSAGE_RIEN_ENTENDU
    assert not r.a_trouve
    assert r.score is None
    assert r.latences.recherche_s == 0.0
    assert textes_dits == [MESSAGE_RIEN_ENTENDU]  # le message est quand même dit à voix haute


def test_la_recherche_recoit_la_transcription(faux_pipeline):
    appels = faux_pipeline("euh c'est quoi le MTCN")
    pipeline.repondre("peu_importe.wav")
    assert appels == ["euh c'est quoi le MTCN"]


def test_latences_coherentes(faux_pipeline):
    faux_pipeline("une question")
    lat = pipeline.repondre("peu_importe.wav").latences
    assert min(lat.asr_s, lat.recherche_s, lat.tts_s) >= 0
    assert lat.total_s >= max(lat.asr_s, lat.recherche_s, lat.tts_s)


def test_vers_ligne_a_plat(faux_pipeline):
    faux_pipeline("une question")
    ligne = pipeline.repondre("peu_importe.wav").vers_ligne()
    assert {"transcription", "id_question", "score", "asr_s", "total_s"} <= set(ligne)
    assert isinstance(ligne["audio_reponse"], str)
    assert not any(isinstance(v, dict) for v in ligne.values())


def test_reponse_non_modifiable(faux_pipeline):
    faux_pipeline("une question")
    r = pipeline.repondre("peu_importe.wav")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.transcription = "modifiée"


def test_prechauffer_prepare_toutes_les_voix(monkeypatch):
    """Les 30 réponses vocales et les 2 messages fixes sont mis en cache au démarrage."""
    textes = []
    monkeypatch.setattr(pipeline.asr, "prechauffer", lambda: None)
    monkeypatch.setattr(
        pipeline, "charger_moteur", lambda: type("M", (), {"faq": list(FAQ.values())})
    )
    monkeypatch.setattr(pipeline, "prechauffer_voix", textes.extend)
    pipeline.prechauffer()
    assert {e.reponse_vocale for e in FAQ.values()} <= set(textes)
    assert {MESSAGE_JE_NE_SAIS_PAS, MESSAGE_RIEN_ENTENDU} <= set(textes)
