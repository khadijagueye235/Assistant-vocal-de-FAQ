"""Constantes partagées par tout le projet : un seul endroit à modifier."""
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
FAQ_CSV = RACINE / "data" / "faq.csv"
AUDIO_TEST_DIR = RACINE / "data" / "audio_test"
RESULTATS_DIR = RACINE / "results"

# --- ASR ---
WHISPER_MODELE = "small"   # à comparer plus tard avec "medium"
LANGUE = "fr"

# --- Pipeline ---
# Dit à voix haute quand le VAD n'a détecté aucune parole.
MESSAGE_RIEN_ENTENDU = "Je n'ai rien entendu. Pouvez-vous répéter votre question ?"
