"""Validation du jeu de développement data/dev/questions_dev.csv."""

import collections

from scripts.evaluer_dev_audio import erreurs_mots
from scripts.evaluer_recherche import charger_dev, nettoyer


def test_chaque_question_de_la_faq_a_au_moins_3_reformulations():
    compte = collections.Counter(q.id_attendu for q in charger_dev() if q.id_attendu)
    assert set(compte) == set(range(1, 31))
    assert min(compte.values()) >= 3


def test_questions_hors_faq_presentes():
    hors = [q for q in charger_dev() if q.id_attendu is None]
    assert len(hors) >= 10


def test_aucune_question_vide_ni_en_double():
    textes = [q.texte.strip().lower() for q in charger_dev()]
    assert all(textes)
    assert len(set(textes)) == len(textes)


def test_nettoyage_retire_les_hesitations_seulement():
    assert nettoyer("euh bon je veux  heu retirer") == "bon je veux retirer"
    assert nettoyer("le bon numéro") == "le bon numéro"


def test_erreurs_mots_ignore_casse_ponctuation_et_hesitations():
    assert erreurs_mots("Euh, c'est quoi le MTCN ?", "c'est quoi le mtcn") == (0, 5)


def test_erreurs_mots_compte_substitution_suppression_insertion():
    assert erreurs_mots("je veux retirer mon argent", "je veux retiré mon argent") == (1, 5)
    assert erreurs_mots("je veux retirer mon argent", "je veux retirer argent") == (1, 5)
    assert erreurs_mots("je veux retirer", "alors je veux retirer") == (1, 3)
    assert erreurs_mots("je veux retirer", "") == (3, 3)
