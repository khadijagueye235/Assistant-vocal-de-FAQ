"""Synthèse vocale : logique testée sans réseau (Edge TTS remplacé), plus un test réel."""

import pytest
from conftest import besoin_ffmpeg

from src import tts
from src.asr import lire_audio


class FauxCommunicate:
    """Remplace edge_tts.Communicate : enregistre les paramètres, écrit un faux MP3."""

    appels: list[dict] = []
    contenu = b"faux mp3"
    erreur: Exception | None = None

    def __init__(self, texte, voix, **options):
        FauxCommunicate.appels.append({"texte": texte, "voix": voix, **options})

    def save_sync(self, chemin):
        with open(chemin, "wb") as fichier:
            fichier.write(self.contenu)  # écrit avant l'erreur : simule un fichier partiel
        if self.erreur:
            raise self.erreur


@pytest.fixture
def faux_edge(monkeypatch):
    FauxCommunicate.appels = []
    FauxCommunicate.contenu = b"faux mp3"
    FauxCommunicate.erreur = None
    monkeypatch.setattr(tts.edge_tts, "Communicate", FauxCommunicate)
    return FauxCommunicate


def test_ecrit_un_fichier(faux_edge, tmp_path):
    chemin = tts.parler("Bonjour", dossier=tmp_path)
    assert chemin.parent == tmp_path
    assert chemin.suffix == ".mp3"
    assert chemin.read_bytes() == b"faux mp3"


def test_parametres_transmis(faux_edge, tmp_path):
    tts.parler("Bonjour", voix="fr-FR-HenriNeural", debit="-10%", dossier=tmp_path)
    appel = faux_edge.appels[0]
    assert appel["texte"] == "Bonjour"
    assert appel["voix"] == "fr-FR-HenriNeural"
    assert appel["rate"] == "-10%"
    assert appel["receive_timeout"] == tts.TTS_DELAI_S  # jamais l'attente de 60 s par défaut


def test_un_fichier_different_par_appel(faux_edge, tmp_path):
    assert tts.parler("Un", dossier=tmp_path) != tts.parler("Deux", dossier=tmp_path)


@pytest.mark.parametrize("texte", ["", "   "])
def test_texte_vide_refuse(faux_edge, tmp_path, texte):
    with pytest.raises(ValueError):
        tts.parler(texte, dossier=tmp_path)
    assert faux_edge.appels == []


def test_panne_reseau_signalee_sans_fichier_partiel(faux_edge, tmp_path):
    faux_edge.erreur = ConnectionError("réseau coupé")
    with pytest.raises(tts.ErreurTTS, match="ConnectionError"):
        tts.parler("Bonjour", dossier=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_aucun_audio_recu(faux_edge, tmp_path):
    faux_edge.contenu = b""
    with pytest.raises(tts.ErreurTTS, match="aucun audio"):
        tts.parler("Bonjour", dossier=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cache_meme_texte_un_seul_appel_reseau(faux_edge, tmp_path):
    premier = tts.parler("Bonjour", dossier=tmp_path)
    second = tts.parler("Bonjour", dossier=tmp_path)
    assert premier == second
    assert len(faux_edge.appels) == 1  # le second appel est servi par le cache


def test_cache_distingue_voix_et_debit(faux_edge, tmp_path):
    a = tts.parler("Bonjour", voix="fr-FR-DeniseNeural", dossier=tmp_path)
    b = tts.parler("Bonjour", voix="fr-FR-HenriNeural", dossier=tmp_path)
    c = tts.parler("Bonjour", voix="fr-FR-DeniseNeural", debit="-10%", dossier=tmp_path)
    assert len({a, b, c}) == 3


def test_sans_cache_chaque_appel_synthetise(faux_edge, tmp_path):
    a = tts.parler("Bonjour", dossier=tmp_path, cache=False)
    b = tts.parler("Bonjour", dossier=tmp_path, cache=False)
    assert a != b
    assert len(faux_edge.appels) == 2


def test_panne_ne_met_rien_en_cache(faux_edge, tmp_path):
    faux_edge.erreur = ConnectionError("réseau coupé")
    with pytest.raises(tts.ErreurTTS):
        tts.parler("Bonjour", dossier=tmp_path)
    faux_edge.erreur = None
    tts.parler("Bonjour", dossier=tmp_path)  # le réseau revient : on resynthétise
    assert len(faux_edge.appels) == 2
    assert len(list(tmp_path.iterdir())) == 1  # aucun fichier temporaire oublié


def test_prechauffer_voix_remplit_le_cache_sans_doublon(faux_edge, tmp_path):
    assert tts.prechauffer_voix(["Un", "Deux", "Un", "  "], dossier=tmp_path) == 2
    assert len(faux_edge.appels) == 2
    tts.parler("Deux", dossier=tmp_path)
    assert len(faux_edge.appels) == 2  # déjà en cache


def test_prechauffer_voix_ne_plante_pas_sans_reseau(faux_edge, tmp_path, caplog):
    faux_edge.erreur = ConnectionError("réseau coupé")
    assert tts.prechauffer_voix(["Un", "Deux"], dossier=tmp_path) == 0
    assert "incomplet" in caplog.text


@pytest.mark.reseau
@besoin_ffmpeg
def test_synthese_reelle(tmp_path):
    chemin = tts.parler("Bonjour, je suis l'assistant Western Union.", dossier=tmp_path)
    duree_s = len(lire_audio(chemin)) / 16_000
    assert 1.0 < duree_s < 10.0  # une vraie phrase lue, pas du silence ni un fichier vide
