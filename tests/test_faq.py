"""Validation de data/faq.csv : la base de connaissances doit rester saine."""

import csv

import pytest

from src.config import FAQ_CSV

COLONNES_OBLIGATOIRES = [
    "id",
    "categorie",
    "question",
    "reponse",
    "url_source",
    "question_recherche",
    "reponse_vocale",
]


@pytest.fixture(scope="module")
def faq() -> list[dict[str, str]]:
    with open(FAQ_CSV, encoding="utf-8", newline="") as fichier:
        return list(csv.DictReader(fichier))


def test_trente_questions(faq):
    assert len(faq) == 30


def test_colonnes_presentes(faq):
    manquantes = set(COLONNES_OBLIGATOIRES) - set(faq[0])
    assert not manquantes, f"Colonnes manquantes : {manquantes}"


def test_ids_de_1_a_30_sans_doublon(faq):
    assert sorted(int(ligne["id"]) for ligne in faq) == list(range(1, 31))


def test_aucun_champ_obligatoire_vide(faq):
    vides = [
        (ligne["id"], colonne)
        for ligne in faq
        for colonne in COLONNES_OBLIGATOIRES
        if not ligne[colonne].strip()
    ]
    assert not vides, f"Champs vides (id, colonne) : {vides}"


def test_sources_publiques_western_union(faq):
    for ligne in faq:
        assert ligne["url_source"].startswith("https://www.westernunion.com/"), ligne["id"]


@pytest.mark.parametrize("colonne", ["question", "question_recherche"])
def test_questions_sans_doublon(faq, colonne):
    questions = [ligne[colonne].strip().lower() for ligne in faq]
    assert len(set(questions)) == len(questions)
