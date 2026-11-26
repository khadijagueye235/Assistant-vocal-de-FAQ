"""Synthèse vocale (TTS) : texte -> fichier audio, avec Edge TTS.

Edge TTS est le service de lecture à voix haute de Microsoft Edge : la synthèse
se fait sur les serveurs de Microsoft, pas sur la machine. Il faut donc une
connexion Internet, et tout échec réseau est signalé par `ErreurTTS` pour que
le pipeline puisse continuer sans la voix (la réponse écrite reste affichée).

La bibliothèque edge-tts est asynchrone (asyncio). On utilise `save_sync`, qui
exécute la synthèse dans un fil d'exécution séparé avec sa propre boucle
d'événements : l'appel fonctionne aussi dans Colab et Gradio, où une boucle
asyncio tourne déjà (un simple asyncio.run() y lèverait
« This event loop is already running »).
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import edge_tts

from src.config import TTS_DEBIT, TTS_DELAI_S, TTS_VOIX

DOSSIER_AUDIO_SORTIE = Path(tempfile.gettempdir()) / "assistant_faq"


class ErreurTTS(RuntimeError):
    """La synthèse vocale a échoué (réseau coupé, service indisponible, voix inconnue...)."""


def parler(
    texte: str,
    voix: str = TTS_VOIX,
    debit: str = TTS_DEBIT,
    dossier: Path = DOSSIER_AUDIO_SORTIE,
) -> Path:
    """Synthétise `texte` et renvoie le chemin du fichier MP3 produit.

    Chaque appel écrit un nouveau fichier (nom unique) : deux utilisateurs de
    l'interface ne s'écrasent pas leurs réponses.
    Lève ValueError si le texte est vide, ErreurTTS si la synthèse échoue.
    """
    if not texte.strip():
        raise ValueError("Rien à dire : le texte est vide.")

    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"reponse_{uuid.uuid4().hex[:8]}.mp3"
    communication = edge_tts.Communicate(
        texte,
        voix,
        rate=debit,
        # Par défaut, edge-tts peut attendre 60 s une réponse : bien trop pour
        # un assistant vocal. Au-delà de TTS_DELAI_S, on abandonne la voix.
        connect_timeout=TTS_DELAI_S,
        receive_timeout=TTS_DELAI_S,
    )
    try:
        communication.save_sync(str(chemin))
    except Exception as erreur:  # réseau, service, voix : toutes les causes se valent ici
        chemin.unlink(missing_ok=True)  # pas de fichier à moitié écrit
        cause = f"{type(erreur).__name__}: {erreur}"
        raise ErreurTTS(f"Synthèse vocale impossible ({cause})") from erreur

    if not chemin.is_file() or chemin.stat().st_size == 0:
        chemin.unlink(missing_ok=True)
        raise ErreurTTS("Synthèse vocale impossible : aucun audio reçu.")
    return chemin
