"""Réglages du projet : un seul endroit à modifier.

Chaque réglage a une valeur par défaut et peut être remplacé par une variable
d'environnement du même nom, sans toucher au code. Exemple dans Colab :

    import os
    os.environ["WHISPER_MODELE"] = "medium"   # AVANT d'importer src

Utile pour comparer des variantes (modèle, seuil...) lors des mesures.
"""

from __future__ import annotations

import os
from pathlib import Path


def _texte(nom: str, defaut: str) -> str:
    return os.environ.get(nom, defaut)


def _entier(nom: str, defaut: int) -> int:
    return int(os.environ.get(nom, defaut))


def _decimal(nom: str, defaut: float) -> float:
    return float(os.environ.get(nom, defaut))


def _booleen(nom: str, defaut: bool) -> bool:
    valeur = os.environ.get(nom)
    if valeur is None:
        return defaut
    return valeur.strip().lower() in {"1", "true", "oui", "vrai"}


# --- Chemins ---
RACINE = Path(__file__).resolve().parent.parent
FAQ_CSV = RACINE / "data" / "faq.csv"
AUDIO_TEST_DIR = RACINE / "data" / "audio_test"
RESULTATS_DIR = RACINE / "results"

# --- ASR (faster-whisper) ---
WHISPER_MODELE = _texte("WHISPER_MODELE", "small")  # à comparer avec "medium"
LANGUE = _texte("LANGUE", "fr")
ASR_BEAM_SIZE = _entier("ASR_BEAM_SIZE", 5)  # 1 = plus rapide, 5 = plus précis
# Silence (ms) à partir duquel le VAD coupe un segment de parole.
ASR_VAD_SILENCE_MS = _entier("ASR_VAD_SILENCE_MS", 500)
# Vocabulaire métier soufflé à Whisper ; vide = aucun (sera réglé dans une brique dédiée).
ASR_PROMPT_INITIAL = _texte("ASR_PROMPT_INITIAL", "") or None

# --- Recherche (embeddings) ---
# Choix justifié par scripts/evaluer_recherche.py sur data/dev (résultats dans
# results/evaluation_recherche.csv) : meilleur top1 et meilleure séparation
# FAQ / hors FAQ à latence égale avec MiniLM ; e5-base n'est pas
# significativement meilleur et coûte deux fois plus.
EMBEDDING_MODELE = _texte("EMBEDDING_MODELE", "intfloat/multilingual-e5-small")
# Préfixe ajouté devant chaque texte encodé : "query: " pour la famille e5
# (même préfixe des deux côtés car on compare des questions entre elles).
EMBEDDING_PREFIXE = _texte("EMBEDDING_PREFIXE", "query: ")
# Indexer « question + réponse » : +5 à +8 points de top1 pour tous les modèles testés.
EMBEDDING_AVEC_REPONSE = _booleen("EMBEDDING_AVEC_REPONSE", True)
# Sous ce score, l'assistant répond « je ne sais pas ».
# Calibré par scripts/calibrer_seuil.py sur data/dev (JAMAIS sur le jeu de test).
# 0,87 : meilleur seuil si une erreur coûte autant qu'une bonne réponse, proche
# du meilleur si elle coûte le double, et choisi bas dans le plateau 0,87-0,885
# car les erreurs d'ASR feront baisser les scores réels. Sur le dev : 60/90
# bonnes réponses, 15 refus à tort, 13/15 questions hors FAQ refusées.
# Propre à e5-small : à recalibrer si le modèle change.
SEUIL_SIMILARITE = _decimal("SEUIL_SIMILARITE", 0.87)

# --- TTS (Edge TTS) ---
TTS_VOIX = _texte("TTS_VOIX", "fr-FR-DeniseNeural")

# --- Messages de l'assistant ---
# Dit à voix haute quand le VAD n'a détecté aucune parole.
MESSAGE_RIEN_ENTENDU = "Je n'ai rien entendu. Pouvez-vous répéter votre question ?"
# Dit quand aucune question de la FAQ n'est assez proche.
MESSAGE_JE_NE_SAIS_PAS = (
    "Je ne connais pas la réponse à cette question. "
    "Je vous invite à contacter le service client de Western Union."
)
