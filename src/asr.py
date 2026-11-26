"""ASR : audio -> texte français avec faster-whisper et filtre VAD."""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel

from src.config import (
    ASR_BEAM_SIZE,
    ASR_PROMPT_INITIAL,
    ASR_VAD_SILENCE_MS,
    LANGUE,
    WHISPER_MODELE,
)

logger = logging.getLogger(__name__)

FREQUENCE_HZ = 16_000  # fréquence attendue par Whisper

# Formes d'audio acceptées par transcrire() :
# - chemin de fichier (wav, mp3, m4a...) : cas des fichiers du jeu de test ;
# - tableau numpy : signal mono float32 déjà échantillonné à 16 kHz ;
# - tuple (fréquence, tableau) : format renvoyé par défaut par gr.Audio de Gradio.
EntreeAudio = str | Path | np.ndarray | tuple[int, np.ndarray]


@dataclass(frozen=True)
class Transcription:
    """Résultat d'une transcription. Les durées sont en secondes, non arrondies."""

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
    else:
        # Niveau WARNING : affiché même sans configuration du journal (Colab,
        # scripts). Des latences mesurées sur CPU par erreur fausseraient le rapport.
        logger.warning(
            "Aucun GPU utilisé : Whisper '%s' tourne sur CPU, environ 5 à 20 fois "
            "plus lentement. Sur Colab : Exécution > Modifier le type d'exécution > GPU.",
            nom,
        )
    logger.info("Chargement Whisper '%s' sur %s (%s)", nom, device, compute_type)
    return WhisperModel(nom, device=device, compute_type=compute_type)


def prechauffer(nom: str = WHISPER_MODELE) -> None:
    """Charge le modèle et fait une inférence à vide.

    Le premier appel paie des coûts uniques (initialisation CUDA, allocation
    mémoire). À appeler au démarrage, sinon la première question mesurée
    paraît anormalement lente et fausse la moyenne des latences.
    """
    modele = charger_modele(nom)
    silence = np.zeros(FREQUENCE_HZ, dtype=np.float32)  # 1 s
    # Sans VAD : sinon le silence serait retiré et le modèle ne calculerait rien.
    segments, _ = modele.transcribe(silence, language=LANGUE, vad_filter=False)
    list(segments)  # force le calcul (générateur paresseux)


def _verifier_ffmpeg() -> str:
    """Renvoie le chemin de ffmpeg, ou lève une erreur qui explique quoi installer."""
    chemin = shutil.which("ffmpeg")
    if chemin is None:
        raise RuntimeError(
            "ffmpeg est introuvable. Installez-le : "
            "Colab/Linux « apt install ffmpeg », "
            "Windows « winget install Gyan.FFmpeg.Essentials », "
            "puis ouvrez un nouveau terminal."
        )
    return chemin


def _ffmpeg_vers_16k_mono(arguments_entree: list[str], donnees: bytes | None = None) -> np.ndarray:
    """Fait passer un audio par ffmpeg pour obtenir du mono 16 kHz float32."""
    commande = [_verifier_ffmpeg(), "-loglevel", "error"]
    if donnees is None:
        commande.append("-nostdin")  # ffmpeg ne doit pas lire le clavier
    commande += [*arguments_entree, "-f", "f32le", "-ac", "1", "-ar", str(FREQUENCE_HZ), "-"]
    resultat = subprocess.run(commande, input=donnees, capture_output=True)
    if resultat.returncode != 0:
        erreur = resultat.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg n'a pas pu décoder l'audio : {erreur}")
    return np.frombuffer(resultat.stdout, dtype=np.float32)


def lire_audio(chemin: Path) -> np.ndarray:
    """Décode un fichier audio (wav, mp3, m4a...) en signal mono 16 kHz float32."""
    signal = _ffmpeg_vers_16k_mono(["-i", str(chemin)])
    if signal.size == 0:
        raise ValueError(f"Fichier audio vide : {chemin}")
    return signal


def _depuis_gradio(frequence: int, signal: np.ndarray) -> np.ndarray:
    """Convertit le tuple (fréquence, tableau) de Gradio en mono 16 kHz float32.

    Gradio fournit souvent de l'int16, parfois stéréo (forme (n, 2)), à 44,1 ou
    48 kHz. On normalise dans [-1, 1], on moyenne les canaux, puis ffmpeg
    rééchantillonne (meilleure qualité qu'une interpolation maison).
    """
    if signal.size == 0:
        raise ValueError("Enregistrement vide : aucun son reçu du micro.")
    if np.issubdtype(signal.dtype, np.integer):
        # int16 va de -32768 à 32767 : diviser par 32768 garde tout dans [-1, 1].
        signal = signal.astype(np.float32) / (np.iinfo(signal.dtype).max + 1)
    signal = signal.astype(np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if frequence == FREQUENCE_HZ:
        return signal
    return _ffmpeg_vers_16k_mono(
        ["-f", "f32le", "-ac", "1", "-ar", str(frequence), "-i", "pipe:0"],
        donnees=signal.tobytes(),
    )


def preparer_audio(audio: EntreeAudio) -> np.ndarray:
    """Ramène n'importe quelle entrée acceptée à un signal mono 16 kHz float32."""
    if isinstance(audio, tuple):
        frequence, signal = audio
        return _depuis_gradio(int(frequence), np.asarray(signal))
    if isinstance(audio, np.ndarray):
        if audio.size == 0:
            raise ValueError("Signal audio vide.")
        return audio.astype(np.float32, copy=False)
    chemin = Path(audio)
    if not chemin.is_file():
        raise FileNotFoundError(f"Audio introuvable : {chemin}")
    return lire_audio(chemin)


def transcrire(
    audio: EntreeAudio,
    nom_modele: str = WHISPER_MODELE,
    langue: str = LANGUE,
    beam_size: int = ASR_BEAM_SIZE,
    prompt_initial: str | None = ASR_PROMPT_INITIAL,
) -> Transcription:
    """Transcrit un audio : chemin de fichier, tableau numpy 16 kHz ou tuple Gradio.

    temps_s couvre la préparation de l'audio, le VAD et la transcription,
    donc ce que l'utilisateur attend réellement (hors chargement du modèle).
    """
    modele = charger_modele(nom_modele)

    debut = time.perf_counter()
    signal = preparer_audio(audio)
    segments, _ = modele.transcribe(
        signal,
        language=langue,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": ASR_VAD_SILENCE_MS},
        condition_on_previous_text=False,  # évite les boucles d'hallucination
        initial_prompt=prompt_initial,
    )
    # `segments` est un générateur : le calcul réel se fait pendant cette ligne.
    texte = " ".join(s.text.strip() for s in segments).strip()
    temps = time.perf_counter() - debut

    return Transcription(
        texte=texte,
        duree_audio_s=len(signal) / FREQUENCE_HZ,
        temps_s=temps,
    )
