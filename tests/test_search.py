"""Recherche par embeddings : lecture de la FAQ (rapide) et moteur (charge un modèle)."""

import pytest

from src.search import EntreeFAQ, MoteurRecherche, charger_faq, charger_moteur

# --- Lecture de la FAQ (sans modèle) ---


def test_charger_faq_renvoie_30_entrees_typees():
    faq = charger_faq()
    assert len(faq) == 30
    assert all(isinstance(e, EntreeFAQ) for e in faq)
    assert [e.id for e in faq] == list(range(1, 31))


def test_moteur_refuse_une_faq_vide():
    with pytest.raises(ValueError):
        MoteurRecherche([])


# --- Moteur (charge le modèle d'embeddings) ---


@pytest.fixture(scope="module")
def moteur() -> MoteurRecherche:
    return charger_moteur()


@pytest.mark.modele
def test_chaque_question_se_retrouve_dans_le_top3(moteur):
    """Test de bon sens sur l'index.

    Pas forcément 1re : avec l'index « question + réponse », la réponse dilue
    parfois la question (la n°12 sort derrière la n°4). Ce compromis est
    assumé car l'index Q+R gagne nettement sur les reformulations du jeu de dev.
    """
    for entree in moteur.faq:
        ids = [r.entree.id for r in moteur.classer(entree.question_recherche, k=3)]
        assert entree.id in ids, f"n°{entree.id} absente du top 3 : {ids}"


@pytest.mark.modele
def test_classement_trie_et_de_taille_k(moteur):
    top = moteur.classer("combien coûte un transfert", k=5)
    assert len(top) == 5
    scores = [r.score for r in top]
    assert scores == sorted(scores, reverse=True)
    assert len({r.entree.id for r in top}) == 5


@pytest.mark.modele
@pytest.mark.parametrize(
    "requete, id_attendu",
    [
        ("ça coûte combien pour recevoir de l'argent", 24),
        ("c'est quoi le MTCN là", 10),
        ("jusqu'à combien je peux envoyer par internet", 5),
    ],
)
def test_reformulations_orales_evidentes(moteur, requete, id_attendu):
    assert moteur.chercher(requete).entree.id == id_attendu


@pytest.mark.modele
def test_question_hors_sujet_moins_similaire_qu_une_vraie_question(moteur):
    """Comparaison relative : l'échelle des scores dépend du modèle (e5 : 0,8 à 0,95)."""
    hors_sujet = moteur.chercher("quelle heure est-il").score
    vraie = moteur.chercher("ça coûte combien pour recevoir de l'argent").score
    assert hors_sujet < vraie


@pytest.mark.modele
def test_requete_vide_refusee(moteur):
    with pytest.raises(ValueError):
        moteur.classer("   ")
