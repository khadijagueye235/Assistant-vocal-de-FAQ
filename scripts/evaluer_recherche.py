"""Compare des configurations de recherche sur le jeu de développement.

Usage (depuis la racine du dépôt) :
    python -m scripts.evaluer_recherche

Pour chaque configuration (modèle x texte indexé x nettoyage), mesure :
- top1 : la bonne question est-elle classée 1re ?
- top3 : est-elle dans les 3 premières ?
- MRR  : moyenne de 1/rang de la bonne question (1 = toujours 1re)
- latence médiane d'une requête
- scores des bonnes réponses et des questions hors FAQ (pour le seuil)

Ce jeu sert à CHOISIR et RÉGLER. Il ne remplace pas la mesure finale sur les
enregistrements de B, jamais utilisés pour le réglage.
"""

from __future__ import annotations

import csv
import re
import statistics
import time
from dataclasses import dataclass

from src.config import RACINE, RESULTATS_DIR
from src.search import MoteurRecherche, charger_faq

DEV_CSV = RACINE / "data" / "dev" / "questions_dev.csv"

# (nom court, identifiant Hugging Face, préfixe attendu par le modèle)
MODELES = [
    ("MiniLM-L12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
    ("mpnet-base", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", ""),
    ("e5-small", "intfloat/multilingual-e5-small", "query: "),
    ("e5-base", "intfloat/multilingual-e5-base", "query: "),
]

# Mots d'hésitation retirés par la variante « nettoyée ».
HESITATIONS = re.compile(r"\b(euh+|heu+|hum+|bah|ben)\b", flags=re.IGNORECASE)


def nettoyer(texte: str) -> str:
    return re.sub(r"\s+", " ", HESITATIONS.sub(" ", texte)).strip()


@dataclass
class QuestionDev:
    id_attendu: int | None  # None = hors FAQ
    texte: str
    particularites: str


def charger_dev() -> list[QuestionDev]:
    with open(DEV_CSV, encoding="utf-8", newline="") as fichier:
        return [
            QuestionDev(
                int(ligne["id_attendu"]) if ligne["id_attendu"] else None,
                ligne["question"],
                ligne["particularites"],
            )
            for ligne in csv.DictReader(fichier)
        ]


def evaluer(moteur: MoteurRecherche, dev: list[QuestionDev], nettoyage: bool) -> dict:
    rangs, scores_bons, scores_hors, latences, echecs = [], [], [], [], []
    for q in dev:
        texte = nettoyer(q.texte) if nettoyage else q.texte
        debut = time.perf_counter()
        classement = moteur.classer(texte, k=len(moteur.faq))
        latences.append(time.perf_counter() - debut)
        meilleur = classement[0]
        if q.id_attendu is None:
            scores_hors.append(meilleur.score)
            continue
        rang = next(i for i, r in enumerate(classement, 1) if r.entree.id == q.id_attendu)
        rangs.append(rang)
        if rang == 1:
            scores_bons.append(meilleur.score)
        else:
            echecs.append((q, meilleur.entree.id, rang))
    return {
        "top1": sum(r == 1 for r in rangs) / len(rangs),
        "top3": sum(r <= 3 for r in rangs) / len(rangs),
        "mrr": statistics.mean(1 / r for r in rangs),
        "latence_ms": statistics.median(latences) * 1000,
        "score_bons_min": min(scores_bons),
        "score_bons_mediane": statistics.median(scores_bons),
        "score_hors_mediane": statistics.median(scores_hors),
        "score_hors_max": max(scores_hors),
        "echecs": echecs,
    }


def main() -> None:
    faq, dev = charger_faq(), charger_dev()
    n_faq = sum(q.id_attendu is not None for q in dev)
    print(f"Jeu de dev : {n_faq} questions de la FAQ, {len(dev) - n_faq} hors FAQ\n")

    lignes = []
    for nom, identifiant, prefixe in MODELES:
        for avec_reponse in (False, True):
            moteur = MoteurRecherche(faq, identifiant, prefixe, avec_reponse)
            for nettoyage in (False, True):
                res = evaluer(moteur, dev, nettoyage)
                config = (
                    f"{nom:<11} index={'Q+R' if avec_reponse else 'Q  '} "
                    f"nettoyage={'oui' if nettoyage else 'non'}"
                )
                print(
                    f"{config}  top1={res['top1']:.1%}  top3={res['top3']:.1%}  "
                    f"MRR={res['mrr']:.3f}  {res['latence_ms']:.0f} ms  "
                    f"bons[min={res['score_bons_min']:.2f} méd={res['score_bons_mediane']:.2f}]  "
                    f"hors[méd={res['score_hors_mediane']:.2f} max={res['score_hors_max']:.2f}]"
                )
                echecs = res.pop("echecs")
                lignes.append(
                    {
                        "modele": nom,
                        "index": "Q+R" if avec_reponse else "Q",
                        "nettoyage": nettoyage,
                        **res,
                        "echecs": "; ".join(f"{q.id_attendu}->{trouve}" for q, trouve, _ in echecs),
                    }
                )
    RESULTATS_DIR.mkdir(exist_ok=True)
    sortie = RESULTATS_DIR / "evaluation_recherche.csv"
    with open(sortie, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    print(f"\nRésultats détaillés : {sortie.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
