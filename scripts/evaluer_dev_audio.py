"""Évalue l'ASR et la recherche sur la version audio du jeu de développement.

Usage (depuis la racine du dépôt, après scripts.generer_dev_audio) :
    python -m scripts.evaluer_dev_audio                        # réglages de la config
    python -m scripts.evaluer_dev_audio --prompt "Western Union, MTCN" --label prompt

Pour chaque question, on compare deux chemins :
- TEXTE : la question écrite va directement dans la recherche (ce qu'on a
  calibré jusqu'ici) ;
- AUDIO : la question lue par une voix passe d'abord par Whisper.
L'écart entre les deux mesure ce que coûtent les erreurs de l'ASR.

Mesures :
- TEM (taux d'erreur sur les mots, « WER » en anglais) : nombre minimal de
  mots à remplacer, supprimer ou insérer pour passer de la transcription au
  texte attendu, divisé par le nombre de mots attendus. 0 % = transcription
  parfaite. Les hésitations (« euh ») et la ponctuation sont ignorées.
- top1, et le bilan bonnes / fausses / refus au seuil de la config.

Résultats par question : results/dev_audio_<label>.csv (voix synthétiques,
aucune donnée personnelle : versionnable).
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import statistics

import numpy as np

from scripts.calibrer_seuil import bilan
from scripts.evaluer_recherche import nettoyer
from scripts.generer_dev_audio import DEV_AUDIO_DIR, MANIFESTE_DEV_AUDIO
from src.asr import prechauffer, transcrire
from src.config import ASR_PROMPT_INITIAL, RACINE, RESULTATS_DIR, SEUIL_SIMILARITE, WHISPER_MODELE
from src.search import charger_moteur


def normaliser(texte: str) -> list[str]:
    """Met un texte sous forme de liste de mots comparables (minuscules, sans ponctuation)."""
    texte = nettoyer(texte.lower().replace("’", "'"))
    return re.sub(r"[\W_]+", " ", texte).split()


def erreurs_mots(reference: str, hypothese: str) -> tuple[int, int]:
    """Renvoie (nombre d'erreurs, nombre de mots de la référence).

    Distance d'édition au niveau des mots (programmation dynamique) : on garde
    les deux nombres plutôt que leur rapport pour pouvoir faire un total juste
    sur tout le jeu (une phrase de 3 mots ne doit pas peser autant qu'une de 15).
    """
    ref, hyp = normaliser(reference), normaliser(hypothese)
    ligne = list(range(len(hyp) + 1))  # coût pour passer de "" aux j premiers mots
    for i, mot_ref in enumerate(ref, 1):
        precedente, ligne = ligne, [i] + [0] * len(hyp)
        for j, mot_hyp in enumerate(hyp, 1):
            ligne[j] = min(
                precedente[j] + 1,  # suppression
                ligne[j - 1] + 1,  # insertion
                precedente[j - 1] + (mot_ref != mot_hyp),  # substitution (ou rien)
            )
    return ligne[-1], len(ref)


def _charger_manifeste() -> list[dict]:
    if not MANIFESTE_DEV_AUDIO.is_file():
        raise SystemExit("Jeu audio absent : lancez d'abord  python -m scripts.generer_dev_audio")
    with open(MANIFESTE_DEV_AUDIO, encoding="utf-8", newline="") as fichier:
        return list(csv.DictReader(fichier))


def _bilan_au_seuil(lignes: list[dict], chemin: str) -> dict:
    scores = np.array([lg[f"score_{chemin}"] for lg in lignes])
    justes = np.array([lg[f"id_{chemin}"] == lg["id_attendu"] for lg in lignes])
    dans_faq = np.array([lg["id_attendu"] is not None for lg in lignes])
    return bilan(scores, justes, dans_faq, SEUIL_SIMILARITE)


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument(
        "--prompt", default=ASR_PROMPT_INITIAL, help="vocabulaire soufflé à Whisper"
    )
    parseur.add_argument("--modele", default=WHISPER_MODELE, help="taille du modèle Whisper")
    parseur.add_argument("--label", default="base", help="suffixe du fichier de résultats")
    args = parseur.parse_args()

    manifeste = _charger_manifeste()
    moteur = charger_moteur()
    prechauffer(args.modele)
    print(f"Whisper {args.modele} | prompt : {args.prompt or '(aucun)'}")
    print(f"{len(manifeste)} questions audio, seuil {SEUIL_SIMILARITE}\n")

    lignes = []
    for n, m in enumerate(manifeste, 1):
        t = transcrire(DEV_AUDIO_DIR / m["fichier"], args.modele, prompt_initial=args.prompt)
        texte = moteur.chercher(m["question"])
        audio = moteur.chercher(t.texte) if not t.est_vide else None
        erreurs, n_mots = erreurs_mots(m["question"], t.texte)
        lignes.append(
            {
                "fichier": m["fichier"],
                "id_attendu": int(m["id_attendu"]) if m["id_attendu"] else None,
                "particularites": m["particularites"],
                "voix": m["voix"],
                "question": m["question"],
                "transcription": t.texte,
                "erreurs_mots": erreurs,
                "mots": n_mots,
                "id_texte": texte.entree.id,
                "score_texte": round(texte.score, 4),
                "id_audio": audio.entree.id if audio else None,
                "score_audio": round(audio.score, 4) if audio else 0.0,
                "asr_s": round(t.temps_s, 3),
            }
        )
        print(f"\r{n}/{len(manifeste)}", end="", flush=True)
    print("\n")

    # --- ASR : taux d'erreur sur les mots, global et par particularité ---
    def tem(groupe: list[dict]) -> float:
        return sum(lg["erreurs_mots"] for lg in groupe) / sum(lg["mots"] for lg in groupe)

    par_tag = collections.defaultdict(list)
    for lg in lignes:
        for tag in filter(None, lg["particularites"].split(";")):
            par_tag[tag].append(lg)
    asr_median = statistics.median(lg["asr_s"] for lg in lignes)
    print(f"TEM global : {tem(lignes):.1%}   (ASR médian {asr_median:.2f} s)")
    for tag, groupe in sorted(par_tag.items()):
        print(f"  {tag:<16} {tem(groupe):6.1%}  ({len(groupe)} questions)")

    # --- Recherche : texte écrit vs transcription ---
    faq = [lg for lg in lignes if lg["id_attendu"] is not None]
    for chemin in ("texte", "audio"):
        top1 = sum(lg[f"id_{chemin}"] == lg["id_attendu"] for lg in faq) / len(faq)
        b = _bilan_au_seuil(lignes, chemin)
        print(
            f"\n{chemin.upper():<5} top1 {top1:.1%} | au seuil : {b['bonnes']} bonnes, "
            f"{b['fausses_faq']} fausses + {b['fausses_hors']} hors FAQ acceptées, "
            f"{b['refus_faq']} refus à tort, {b['refus_hors']} hors FAQ refusées "
            f"| utilité {b['utilite']:.0f}"
        )

    perdues = [lg for lg in faq if lg["id_texte"] == lg["id_attendu"] != lg["id_audio"]]
    print(f"\nQuestions perdues à cause de l'ASR ({len(perdues)}) :")
    for lg in perdues:
        print(f"  [{lg['particularites']}] « {lg['question']} »")
        print(f"      -> « {lg['transcription']} »  (trouvé n°{lg['id_audio']})")

    RESULTATS_DIR.mkdir(exist_ok=True)
    sortie = RESULTATS_DIR / f"dev_audio_{args.label}.csv"
    with open(sortie, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    print(f"\nRésultats détaillés : {sortie.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
