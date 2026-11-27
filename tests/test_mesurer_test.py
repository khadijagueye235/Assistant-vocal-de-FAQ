"""Script de mesure du jeu de test oral (sans modèle : repondre() est simulé)."""

import csv
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mesurer_test
from scripts.mesurer_test import _quantile, charger_manifeste, issue, resumer
from src.pipeline import Latences, Reponse

ENTETE = "fichier,id_question_attendue,locuteur,transcription_reference\n"


def test_le_manifeste_versionne_est_valide_et_coherent_avec_les_noms():
    lignes = charger_manifeste()
    assert lignes, "manifeste vide"
    for lg in lignes:
        numero = re.match(r"q(\d+)_", lg["fichier"])
        if numero:  # convention de nommage : qXX_LYY
            assert lg["attendu"] == int(numero.group(1)), lg["fichier"]
        assert re.fullmatch(r"L\d+", lg["locuteur"]), "identifiant anonyme attendu"


def test_manifeste_refuse_doublon_et_id_invalide(tmp_path):
    chemin = tmp_path / "m.csv"
    chemin.write_text(ENTETE + "a.m4a,3,L01,\na.m4a,4,L02,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="double"):
        charger_manifeste(chemin)
    chemin.write_text(ENTETE + "a.m4a,31,L01,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1..30"):
        charger_manifeste(chemin)


def test_manifeste_id_vide_signifie_hors_faq(tmp_path):
    chemin = tmp_path / "m.csv"
    chemin.write_text(ENTETE + "h.m4a,,L04,\n", encoding="utf-8")
    assert charger_manifeste(chemin)[0]["attendu"] is None


def test_issue():
    assert issue(7, 7) == "bonne"
    assert issue(None, None) == "bonne"  # hors FAQ refusée : c'est le bon comportement
    assert issue(7, None) == "refus"
    assert issue(7, 8) == "fausse"
    assert issue(None, 8) == "fausse"  # répondre à une question hors FAQ


def test_quantile_identique_a_numpy():
    valeurs = [0.3, 1.2, 0.5, 0.4, 2.0, 0.35]
    for q in (0.5, 0.95):
        assert _quantile(valeurs, q) == pytest.approx(np.quantile(valeurs, q))


def test_resumer():
    def ligne(attendu, issue_, loc="L01", erreurs="", mots=""):
        lat = dict(asr_s=0.3, recherche_s=0.01, tts_s=0.0, total_s=0.31)
        return dict(attendu=attendu, issue=issue_, locuteur=loc, erreurs_mots=erreurs,
                    mots_reference=mots, **lat)  # fmt: skip

    r = resumer([ligne(1, "bonne", erreurs=1, mots=10), ligne(2, "refus"), ligne(None, "bonne")])
    assert (r["n"], r["bonne"], r["refus"], r["fausse"]) == (3, 2, 1, 0)
    assert (r["bonnes_dans_faq"], r["n_dans_faq"]) == (1, 2)
    assert (r["refus_hors_faq"], r["n_hors_faq"]) == (1, 1)
    assert r["tem"] == pytest.approx(0.1) and r["n_tem"] == 1
    assert r["latences"]["total_s"] == pytest.approx((0.31, 0.31))


@pytest.fixture
def faux_systeme(tmp_path, monkeypatch):
    """Un dossier d'audios factices et un pipeline simulé : 'q01' -> n°1, le reste -> refus."""
    dossier = tmp_path / "audios"
    dossier.mkdir()
    for nom in ("q01_L01.m4a", "q02_L02.m4a", "h01_L04.m4a"):
        (dossier / nom).write_bytes(b"factice")
    manifeste = tmp_path / "manifeste.csv"
    manifeste.write_text(
        ENTETE + "q01_L01.m4a,1,L01,c'est quoi le MTCN\nq02_L02.m4a,2,L02,\n", encoding="utf-8"
    )

    def faux_repondre(chemin):
        trouve = chemin.name.startswith("q01")
        return Reponse(
            transcription="c'est quoi le mtcn" if trouve else "bla bla",
            id_question=1 if trouve else None,
            question_trouvee=None,
            score=0.9 if trouve else 0.8,
            reponse_texte="...",
            audio_reponse=None,
            latences=Latences(0.3, 0.01, 0.0, 0.31),
        )

    import src.pipeline
    import src.search

    monkeypatch.setattr(src.pipeline, "prechauffer", lambda: None)
    monkeypatch.setattr(src.pipeline, "repondre", faux_repondre)
    monkeypatch.setattr(
        src.search, "chercher", lambda texte: SimpleNamespace(entree=SimpleNamespace(id=5))
    )
    sortie = tmp_path / "resultats.csv"

    def lancer(*options):
        argv = ["mesurer_test", "--dossier", str(dossier), "--manifeste", str(manifeste)]
        monkeypatch.setattr(sys, "argv", [*argv, "--sortie", str(sortie), *options])
        mesurer_test.main()
        with open(sortie, encoding="utf-8", newline="") as fichier:
            return list(csv.DictReader(fichier))

    return SimpleNamespace(lancer=lancer, manifeste=manifeste)


def test_mesure_complete(faux_systeme, capsys):
    lignes = faux_systeme.lancer()
    assert [lg["issue"] for lg in lignes] == ["bonne", "refus"]
    assert lignes[0]["erreurs_mots"] == "0" and lignes[1]["erreurs_mots"] == ""
    assert lignes[1]["id_top1"] == "5"
    assert "audio_reponse" not in lignes[0]
    assert "1/2" in capsys.readouterr().out.replace("\r", "\n")


def test_une_seule_mesure_par_audio(faux_systeme):
    faux_systeme.lancer()
    with pytest.raises(SystemExit, match="existe déjà"):
        faux_systeme.lancer()
    with pytest.raises(SystemExit, match="Rien à mesurer"):
        faux_systeme.lancer("--ajouter")


def test_ajouter_ne_mesure_que_les_nouveaux_audios(faux_systeme):
    faux_systeme.lancer()
    with open(faux_systeme.manifeste, "a", encoding="utf-8") as fichier:
        fichier.write("h01_L04.m4a,,L04,\n")
    lignes = faux_systeme.lancer("--ajouter")
    assert [lg["fichier"] for lg in lignes] == ["q01_L01.m4a", "q02_L02.m4a", "h01_L04.m4a"]
    assert lignes[2]["issue"] == "bonne"  # hors FAQ refusée


def test_audio_manquant_arrete_avant_de_charger_les_modeles(faux_systeme, tmp_path):
    (tmp_path / "audios" / "q02_L02.m4a").unlink()
    with pytest.raises(SystemExit, match="introuvable"):
        faux_systeme.lancer()
