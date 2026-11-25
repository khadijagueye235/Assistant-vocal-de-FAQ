"""ASR : audio -> texte français avec faster-whisper et filtre VAD."""
from __future__ import annotations

import ctypes
import glob
import importlib.util
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel

from src.config import LANGUE, WHISPER_MODELE

logger = logging.getLogger(__name__)

FREQUENCE_HZ = 16_000  # fréquence attendue par Whisper


@dataclass(frozen=True)
class Transcription:
    """Résultat d'une transcription."""
    texte: str
    duree_audio_s: float
    temps_s: float

    @property
    def est_vide(self) -> bool:
        """Vrai si le VAD n'a détecté aucune parole."""
        return not self.texte


def _precharger_bibliotheques_cuda() -> None:
    """Charge cuBLAS et cuDNN installés par pip (nvidia-cublas-cu12, nvidia-cudnn-cu12).

    CTranslate2 les cherche dans les chemins système, où elles ne sont pas.
    Deux passes : certaines bibliothèques dépendent d'autres pas encore chargées.
    """
    fichiers: list[str] = []
    for paquet in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            spec = importlib.util.find_spec(paquet)
        except ModuleNotFoundError:
            continue
        for dossier in (spec.submodule_search_locations or []) if spec else []:
            fichiers += sorted(glob.glob(os.path.join(dossier, "lib*.so*")))

    for _ in range(2):
        restants = []
        for fichier in fichiers:
            try:
                ctypes.CDLL(fichier, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                restants.append(fichier)
        fichiers = restants


def _choisir_materiel() -> tuple[str, str]:
    """GPU en float16 si disponible, sinon CPU en int8 (plus lent mais fonctionnel).

    Pour forcer le CPU : os.environ["ASR_FORCER_CPU"] = "1" avant le chargement.
    """
    if os.environ.get("ASR_FORCER_CPU") == "1":
        return "cpu", "int8"
    if ctranslate2.get_cuda_device_count() > 0:
        return "cuda", "float16"
    return "cpu", "int8"


@lru_cache(maxsize=2)
def charger_modele(nom: str = WHISPER_MODELE) -> WhisperModel:
    """Charge le modèle une seule fois par nom (le chargement est coûteux)."""
    device, compute_type = _choisir_materiel()
    if device == "cuda":
        _precharger_bibliotheques_cuda()
    logger.info("Chargement Whisper '%s' sur %s (%s)", nom, device, compute_type)
    return WhisperModel(nom, device=device, compute_type=compute_type)


def lire_audio(chemin: Path) -> np.ndarray:
    """Décode un fichier audio (wav, mp3, m4a...) en signal mono 16 kHz float32, via ffmpeg."""
    commande = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(chemin),
        "-f", "f32le", "-ac", "1", "-ar", str(FREQUENCE_HZ), "-",
    ]
    resultat = subprocess.run(commande, capture_output=True)
    if resultat.returncode != 0:
        erreur = resultat.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg n'a pas pu lire {chemin} : {erreur}")
    signal = np.frombuffer(resultat.stdout, dtype=np.float32)
    if signal.size == 0:
        raise ValueError(f"Fichier audio vide : {chemin}")
    return signal


def transcrire(
    audio: str | Path,
    nom_modele: str = WHISPER_MODELE,
    langue: str = LANGUE,
    beam_size: int = 5,
    prompt_initial: str | None = None,
) -> Transcription:
    """Transcrit un fichier audio (wav, mp3, m4a...).

    temps_s couvre le décodage du fichier, le VAD et la transcription,
    donc ce que l'utilisateur attend réellement.
    """
    chemin = Path(audio)
    if not chemin.is_file():
        raise FileNotFoundError(f"Audio introuvable : {chemin}")

    modele = charger_modele(nom_modele)

    debut = time.perf_counter()
    signal = lire_audio(chemin)
    segments, info = modele.transcribe(
        signal,
        language=langue,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # évite les boucles d'hallucination
        initial_prompt=prompt_initial,
    )
    # `segments` est un générateur : le calcul réel se fait pendant cette ligne.
    texte = " ".join(s.text.strip() for s in segments).strip()
    temps = time.perf_counter() - debut

    return Transcription(
        texte=texte,
        duree_audio_s=round(len(signal) / FREQUENCE_HZ, 2),
        temps_s=round(temps, 2),
    )
