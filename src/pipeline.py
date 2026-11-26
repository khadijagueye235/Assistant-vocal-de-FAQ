"""Assemblage du pipeline : audio -> transcription -> recherche -> réponse -> audio.

`repondre()` est le contrat partagé avec l'interface Gradio (C) et les mesures (B),
décrit dans docs/contrat_repondre.md. Sa signature et le type `Reponse` ne
changeront plus ; seul l'intérieur évoluera.

État actuel : ASR et recherche réels ; le TTS est SIMULÉ en attendant
src/tts.py (champ `simulation=True` dans la réponse).
"""

from __future__ import annotations

import tempfile
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from src import asr
from src.asr import EntreeAudio, transcrire
from src.config import MESSAGE_JE_NE_SAIS_PAS, MESSAGE_RIEN_ENTENDU, SEUIL_SIMILARITE
from src.search import charger_moteur, chercher

DOSSIER_AUDIO_SORTIE = Path(tempfile.gettempdir()) / "assistant_faq"


@dataclass(frozen=True)
class Latences:
    """Durée de chaque étape, en secondes, non arrondie.

    total_s est mesuré de bout en bout : il peut dépasser légèrement la somme
    des étapes (le code entre les étapes prend aussi un peu de temps).
    """

    asr_s: float
    recherche_s: float
    tts_s: float
    total_s: float


@dataclass(frozen=True)
class Reponse:
    """Tout ce que produit une question posée à l'assistant."""

    transcription: str  # ce que l'ASR a compris
    id_question: int | None  # id dans faq.csv ; None = aucune question retenue
    question_trouvee: str | None  # texte de la question de la FAQ retenue
    score: float | None  # similarité de la meilleure candidate ; None si pas de recherche
    reponse_texte: str  # réponse affichée (la version lue épelle les sigles)
    audio_reponse: Path | None  # fichier audio de la réponse parlée
    latences: Latences
    simulation: bool = False  # True tant qu'une étape est simulée

    @property
    def a_trouve(self) -> bool:
        """Vrai si une question de la FAQ a été retenue."""
        return self.id_question is not None

    def vers_ligne(self) -> dict:
        """Aplatit la réponse en dictionnaire simple : une ligne du tableau de mesures."""
        ligne = asdict(self)
        ligne.update(ligne.pop("latences"))  # asr_s, recherche_s, tts_s, total_s
        ligne["audio_reponse"] = str(self.audio_reponse) if self.audio_reponse else ""
        return ligne


def prechauffer() -> None:
    """Charge tous les modèles et fait une inférence à vide.

    À appeler au démarrage (interface, mesures) : sinon la première question
    paie le chargement des modèles et fausse les latences.
    """
    asr.prechauffer()
    charger_moteur()  # encode la FAQ, ce qui fait aussi tourner le modèle une fois


@dataclass(frozen=True)
class _Decision:
    """Résultat de la recherche après application du seuil."""

    id_question: int | None
    question: str | None
    score: float
    texte_affiche: str
    texte_lu: str


def _rechercher(texte: str) -> _Decision:
    """Cherche la question la plus proche et applique le seuil « je ne sais pas »."""
    resultat = chercher(texte)
    if resultat.score < SEUIL_SIMILARITE:
        return _Decision(None, None, resultat.score, MESSAGE_JE_NE_SAIS_PAS, MESSAGE_JE_NE_SAIS_PAS)
    e = resultat.entree
    return _Decision(e.id, e.question, resultat.score, e.reponse, e.reponse_vocale)


# --- Étape provisoire : remplacée par src/tts.py ---


def _parler_simule(texte: str) -> Path:
    """Simule le TTS : écrit 0,5 s de silence dans un fichier WAV."""
    DOSSIER_AUDIO_SORTIE.mkdir(parents=True, exist_ok=True)
    chemin = DOSSIER_AUDIO_SORTIE / f"reponse_{uuid.uuid4().hex[:8]}.wav"
    frequence = 24_000
    with wave.open(str(chemin), "wb") as fichier:
        fichier.setnchannels(1)
        fichier.setsampwidth(2)  # 16 bits
        fichier.setframerate(frequence)
        fichier.writeframes(b"\x00\x00" * (frequence // 2))
    return chemin


def repondre(audio: EntreeAudio) -> Reponse:
    """Répond à une question orale.

    `audio` : chemin de fichier, tableau numpy 16 kHz ou tuple Gradio
    (voir src.asr.transcrire). Appeler prechauffer() au démarrage pour que
    la première mesure de latence ne soit pas faussée.
    """
    debut = time.perf_counter()

    t = time.perf_counter()
    transcription = transcrire(audio)
    asr_s = time.perf_counter() - t

    if transcription.est_vide:
        decision = None
        recherche_s = 0.0
    else:
        t = time.perf_counter()
        decision = _rechercher(transcription.texte)
        recherche_s = time.perf_counter() - t

    t = time.perf_counter()
    audio_reponse = _parler_simule(decision.texte_lu if decision else MESSAGE_RIEN_ENTENDU)
    tts_s = time.perf_counter() - t

    return Reponse(
        transcription=transcription.texte,
        id_question=decision.id_question if decision else None,
        question_trouvee=decision.question if decision else None,
        score=decision.score if decision else None,
        reponse_texte=decision.texte_affiche if decision else MESSAGE_RIEN_ENTENDU,
        audio_reponse=audio_reponse,
        latences=Latences(asr_s, recherche_s, tts_s, time.perf_counter() - debut),
        simulation=True,
    )
