"""Calibre le seuil « je ne sais pas » sur le jeu de développement.

Usage (depuis la racine du dépôt) :
    python -m scripts.calibrer_seuil

Avec le modèle de la config, chaque question du jeu de dev reçoit le score de
sa meilleure candidate. Pour un seuil donné, l'assistant :
- RÉPOND si score >= seuil (bonne ou mauvaise réponse) ;
- dit « je ne sais pas » sinon.

Une réponse fausse dite avec assurance est pire pour un service client qu'un
« je ne sais pas » qui renvoie vers un conseiller. On choisit donc le seuil
qui maximise l'utilité :
    utilité = bonnes réponses - COUT_ERREUR x réponses fausses
(un refus vaut 0 ; une réponse à une question hors FAQ est une réponse fausse).

Le maximum brut n'est pas forcément le bon choix : la courbe est bruitée
(15 questions hors FAQ seulement). On préfère un seuil au milieu d'un plateau,
stable quand on change le coût, et plutôt bas car les vraies transcriptions
(erreurs d'ASR) feront baisser les scores. Le choix final est dans src/config.py.

On ne calibre JAMAIS sur le jeu de test de B : le score final serait truqué.
"""

from __future__ import annotations

import numpy as np

from scripts.evaluer_recherche import charger_dev
from src.config import EMBEDDING_MODELE
from src.search import charger_moteur

COUT_ERREUR = 2.0  # une réponse fausse coûte deux fois ce que rapporte une bonne


def bilan(scores: np.ndarray, justes: np.ndarray, dans_faq: np.ndarray, seuil: float) -> dict:
    """Compte les issues pour un seuil donné."""
    repond = scores >= seuil
    bonnes = int(np.sum(repond & dans_faq & justes))
    fausses = int(np.sum(repond & ~(dans_faq & justes)))
    return {
        "seuil": seuil,
        "bonnes": bonnes,
        "fausses_faq": int(np.sum(repond & dans_faq & ~justes)),
        "fausses_hors": int(np.sum(repond & ~dans_faq)),
        "refus_faq": int(np.sum(~repond & dans_faq)),
        "refus_hors": int(np.sum(~repond & ~dans_faq)),
        "utilite": bonnes - COUT_ERREUR * fausses,
        "utilite_cout1": bonnes - fausses,  # pour vérifier la sensibilité au coût choisi
    }


def main() -> None:
    moteur, dev = charger_moteur(), charger_dev()
    resultats = [moteur.chercher(q.texte) for q in dev]
    scores = np.array([r.score for r in resultats])
    dans_faq = np.array([q.id_attendu is not None for q in dev])
    justes = np.array([r.entree.id == q.id_attendu for r, q in zip(resultats, dev, strict=True)])

    print(f"Modèle : {EMBEDDING_MODELE}")
    print(f"Jeu de dev : {dans_faq.sum()} questions de la FAQ, {(~dans_faq).sum()} hors FAQ")
    print(f"Coût d'une réponse fausse : {COUT_ERREUR} x une bonne réponse\n")
    print(" seuil | bonnes | fausses (FAQ / hors) | refus (FAQ / hors) | utilité | (coût 1)")

    seuils = np.round(np.arange(0.80, 0.921, 0.005), 3)
    bilans = [bilan(scores, justes, dans_faq, s) for s in seuils]
    meilleur = max(bilans, key=lambda b: b["utilite"])
    for b in bilans:
        marque = "  <- meilleur" if b is meilleur else ""
        print(
            f" {b['seuil']:.3f} | {b['bonnes']:6d} "
            f"| {b['fausses_faq']:9d} / {b['fausses_hors']:<4d}"
            f"     | {b['refus_faq']:8d} / {b['refus_hors']:<4d}     | {b['utilite']:6.1f}"
            f"  | {b['utilite_cout1']:6.1f}{marque}"
        )

    sans_seuil = bilan(scores, justes, dans_faq, -1.0)
    print(
        f"\nSans seuil : {sans_seuil['bonnes']} bonnes, "
        f"{sans_seuil['fausses_faq'] + sans_seuil['fausses_hors']} fausses, "
        f"utilité {sans_seuil['utilite']:.1f}"
    )
    print(
        f"Maximum d'utilité : {meilleur['seuil']:.3f} -> {meilleur['bonnes']} bonnes, "
        f"{meilleur['fausses_faq'] + meilleur['fausses_hors']} fausses, "
        f"{meilleur['refus_faq'] + meilleur['refus_hors']} refus, "
        f"utilité {meilleur['utilite']:.1f}"
    )


if __name__ == "__main__":
    main()
