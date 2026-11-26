"""Recherche : trouve la question de la FAQ la plus proche par embeddings.

Principe :
1. Au démarrage, chaque question de la FAQ est transformée en vecteur
   (embedding) par un modèle de langue. Fait une seule fois.
2. À chaque requête, la question transcrite est transformée en vecteur
   avec le même modèle.
3. On compare ce vecteur aux 30 vecteurs de la FAQ par similarité cosinus ;
   la plus haute similarité désigne la question retenue.

Les vecteurs sont normalisés (longueur 1) : la similarité cosinus se réduit
alors à un simple produit scalaire, très rapide.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODELE, FAQ_CSV

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntreeFAQ:
    """Une ligne de data/faq.csv."""

    id: int
    categorie: str
    question: str  # texte d'origine, affiché à l'utilisateur
    question_recherche: str  # version sans ambiguïté, utilisée pour les embeddings
    reponse: str  # réponse affichée
    reponse_vocale: str  # réponse adaptée à la lecture à voix haute
    url_source: str


@dataclass(frozen=True)
class Resultat:
    """Une question candidate et sa similarité avec la requête."""

    entree: EntreeFAQ
    score: float  # similarité cosinus : 1 = même sens, vers 0 = sans rapport


def charger_faq(chemin: Path = FAQ_CSV) -> list[EntreeFAQ]:
    """Lit la FAQ et renvoie ses entrées dans l'ordre du fichier."""
    with open(chemin, encoding="utf-8", newline="") as fichier:
        return [
            EntreeFAQ(
                id=int(ligne["id"]),
                categorie=ligne["categorie"],
                question=ligne["question"],
                question_recherche=ligne["question_recherche"],
                reponse=ligne["reponse"],
                reponse_vocale=ligne["reponse_vocale"],
                url_source=ligne["url_source"],
            )
            for ligne in csv.DictReader(fichier)
        ]


class MoteurRecherche:
    """Index des questions de la FAQ, prêt à être interrogé."""

    def __init__(
        self,
        faq: list[EntreeFAQ],
        nom_modele: str = EMBEDDING_MODELE,
        prefixe_requete: str = "",
    ) -> None:
        """Charge le modèle et encode les questions de la FAQ (coûteux, fait une fois).

        prefixe_requete : certains modèles (famille e5) attendent un préfixe
        comme "query: " devant chaque texte ; vide pour les autres.
        """
        if not faq:
            raise ValueError("La FAQ est vide.")
        self.faq = faq
        self.nom_modele = nom_modele
        self.prefixe_requete = prefixe_requete
        logger.info("Chargement du modèle d'embeddings '%s'", nom_modele)
        self.modele = SentenceTransformer(nom_modele)
        # Matrice (30, dimension) : une ligne par question de la FAQ.
        self.vecteurs_faq = self._encoder([e.question_recherche for e in faq])

    def _encoder(self, textes: list[str]) -> np.ndarray:
        """Transforme des textes en vecteurs normalisés (longueur 1)."""
        return self.modele.encode(
            [self.prefixe_requete + t for t in textes],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def classer(self, texte: str, k: int = 3) -> list[Resultat]:
        """Renvoie les k questions les plus proches, de la plus à la moins similaire.

        Le classement complet sert à l'analyse des échecs : savoir si la bonne
        question était 2e ou absente du haut du classement.
        """
        if not texte.strip():
            raise ValueError("Texte de requête vide.")
        vecteur = self._encoder([texte])[0]
        scores = self.vecteurs_faq @ vecteur  # produit scalaire = cosinus ici
        meilleurs = np.argsort(scores)[::-1][:k]
        return [Resultat(self.faq[i], float(scores[i])) for i in meilleurs]

    def chercher(self, texte: str) -> Resultat:
        """Renvoie la question la plus proche (sans seuil : décidé par l'appelant)."""
        return self.classer(texte, k=1)[0]


@lru_cache(maxsize=1)
def charger_moteur() -> MoteurRecherche:
    """Moteur par défaut (FAQ et modèle de la config), créé une seule fois."""
    return MoteurRecherche(charger_faq())


def chercher(texte: str) -> Resultat:
    """Raccourci : question la plus proche avec le moteur par défaut."""
    return charger_moteur().chercher(texte)
