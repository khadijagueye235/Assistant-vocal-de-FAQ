"""Validation du jeu de développement data/dev/questions_dev.csv."""

import collections

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
