"""Fabrique la version audio du jeu de développement, avec des voix synthétiques.

Usage (depuis la racine du dépôt, connexion Internet nécessaire) :
    python -m scripts.generer_dev_audio            # ne refait pas les fichiers existants
    python -m scripts.generer_dev_audio --refaire  # régénère tout

Chaque question de data/dev/questions_dev.csv est lue par une voix Edge TTS.
On fait tourner 8 voix (France, Belgique, Suisse, Canada ; femmes et hommes)
et 3 débits, pour que le jeu ne repose pas sur une seule voix. L'attribution
est fixe (question n°i -> voix i mod 8, débit i mod 3) : deux personnes qui
lancent le script obtiennent les mêmes fichiers, et le jeu reste reproductible.

Limite à garder en tête : une voix synthétique articule mieux qu'un humain
qui hésite, et prononce mal le wolof. Ce jeu sert à COMPARER des variantes
(prompt de l'ASR, seuil...), pas à annoncer le score final : celui-ci vient
des enregistrements de B.

Les fichiers audio ne sont pas versionnés (voir .gitignore) : on les régénère.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

from scripts.evaluer_recherche import charger_dev
from src.config import RACINE
from src.tts import ErreurTTS, parler

DEV_AUDIO_DIR = RACINE / "data" / "dev_audio"
MANIFESTE_DEV_AUDIO = DEV_AUDIO_DIR / "manifeste.csv"

VOIX = [
    "fr-FR-HenriNeural",
    "fr-FR-DeniseNeural",
    "fr-BE-GerardNeural",
    "fr-FR-EloiseNeural",
    "fr-CH-FabriceNeural",
    "fr-BE-CharlineNeural",
    "fr-CA-AntoineNeural",
    "fr-CH-ArianeNeural",
]
DEBITS = ["+0%", "-10%", "+10%"]
ESSAIS = 3  # un service en ligne peut échouer ponctuellement : on réessaie


def _synthetiser(texte: str, voix: str, debit: str, nom: str) -> None:
    """Synthétise vers DEV_AUDIO_DIR/nom, en réessayant en cas d'échec réseau."""
    for essai in range(1, ESSAIS + 1):
        try:
            chemin = parler(texte, voix=voix, debit=debit, dossier=DEV_AUDIO_DIR)
            chemin.replace(DEV_AUDIO_DIR / nom)
            return
        except ErreurTTS:
            if essai == ESSAIS:
                raise
            time.sleep(2 * essai)


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--refaire", action="store_true", help="régénère les fichiers existants")
    refaire = parseur.parse_args().refaire

    dev = charger_dev()
    DEV_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    lignes, crees = [], 0
    for i, q in enumerate(dev):
        voix, debit = VOIX[i % len(VOIX)], DEBITS[i % len(DEBITS)]
        nom = f"dev_{i:03d}.mp3"
        if refaire or not (DEV_AUDIO_DIR / nom).is_file():
            _synthetiser(q.texte, voix, debit, nom)
            crees += 1
            print(f"\r{i + 1}/{len(dev)} {nom}", end="", file=sys.stderr)
        lignes.append(
            {
                "fichier": nom,
                "id_attendu": q.id_attendu if q.id_attendu is not None else "",
                "question": q.texte,
                "particularites": q.particularites,
                "voix": voix,
                "debit": debit,
            }
        )

    with open(MANIFESTE_DEV_AUDIO, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    print(f"\n{crees} fichiers créés, {len(lignes) - crees} déjà présents -> {DEV_AUDIO_DIR}")


if __name__ == "__main__":
    main()
