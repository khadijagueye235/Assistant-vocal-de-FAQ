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

Cache : les réponses de la FAQ sont connues à l'avance, donc chaque texte n'est
synthétisé qu'une fois. Le fichier est nommé d'après une empreinte (voix, débit,
texte) : la même demande renvoie le même fichier, en quelques millisecondes,
au lieu d'environ 1 s de requête réseau. `prechauffer_voix` remplit le cache au
démarrage. Mesure qui a conduit à ce choix (et écarté la synthèse phrase par
phrase) : docs/latence_tts.md.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts

from src.config import TTS_CACHE, TTS_DEBIT, TTS_DELAI_S, TTS_VOIX

DOSSIER_AUDIO_SORTIE = Path(tempfile.gettempdir()) / "assistant_faq"
REQUETES_SIMULTANEES = 8  # pour le préchauffage : assez pour aller vite, sans abuser du service

journal = logging.getLogger(__name__)


class ErreurTTS(RuntimeError):
    """La synthèse vocale a échoué (réseau coupé, service indisponible, voix inconnue...)."""


def _nom_en_cache(texte: str, voix: str, debit: str) -> str:
    """Nom de fichier déterminé par le contenu : même voix, débit et texte -> même nom."""
    empreinte = hashlib.sha256(f"{voix}|{debit}|{texte}".encode()).hexdigest()[:16]
    return f"tts_{empreinte}.mp3"


def _synthetiser(texte: str, voix: str, debit: str, chemin: Path) -> None:
    """Écrit la synthèse dans `chemin` ; lève ErreurTTS sans laisser de fichier partiel."""
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


def parler(
    texte: str,
    voix: str = TTS_VOIX,
    debit: str = TTS_DEBIT,
    dossier: Path = DOSSIER_AUDIO_SORTIE,
    cache: bool = TTS_CACHE,
) -> Path:
    """Synthétise `texte` et renvoie le chemin du fichier MP3.

    Avec le cache, un texte déjà dit renvoie le fichier existant (à ne pas
    modifier ni supprimer : il est partagé). Sans cache, chaque appel écrit un
    nouveau fichier.
    Lève ValueError si le texte est vide, ErreurTTS si la synthèse échoue.
    """
    if not texte.strip():
        raise ValueError("Rien à dire : le texte est vide.")
    dossier.mkdir(parents=True, exist_ok=True)

    if not cache:
        chemin = dossier / f"reponse_{uuid.uuid4().hex[:8]}.mp3"
        _synthetiser(texte, voix, debit, chemin)
        return chemin

    chemin = dossier / _nom_en_cache(texte, voix, debit)
    if chemin.is_file() and chemin.stat().st_size > 0:
        return chemin
    # On écrit d'abord sous un nom temporaire, puis on renomme : le renommage
    # est atomique, donc un autre utilisateur ne lit jamais un fichier à moitié écrit.
    temporaire = dossier / f".{chemin.stem}_{uuid.uuid4().hex[:8]}.part"
    _synthetiser(texte, voix, debit, temporaire)
    temporaire.replace(chemin)
    return chemin


def prechauffer_voix(textes: Iterable[str], **options) -> int:
    """Remplit le cache pour tous ces textes, en parallèle ; renvoie le nombre de réussites.

    Ne lève jamais d'erreur : sans réseau au démarrage, l'assistant doit quand
    même démarrer (les réponses manquantes seront synthétisées à la demande).
    """
    textes = list(dict.fromkeys(t for t in textes if t.strip()))  # sans doublons

    def essayer(texte: str) -> bool:
        try:
            parler(texte, cache=True, **options)
            return True
        except ErreurTTS as erreur:
            journal.warning("Préchauffage de la voix incomplet : %s", erreur)
            return False

    with ThreadPoolExecutor(max_workers=REQUETES_SIMULTANEES) as executeur:
        return sum(executeur.map(essayer, textes))
