"""Mesure finale du système sur le jeu de test oral (rôle B).

Usage (depuis la racine du dépôt, de préférence sur GPU) :
    python -m scripts.mesurer_test                                # audios dans data/audio_test/
    python -m scripts.mesurer_test --dossier /content/drive/MyDrive/audios_test

Règles (voir data/audio_test/LISEZMOI.md) :
- le manifeste est rempli AVANT la mesure : la question attendue ne doit
  jamais être choisie après avoir vu la réponse du système ;
- chaque audio n'est mesuré qu'une fois. Le script refuse d'écraser un
  résultat existant (--refaire pour forcer, à justifier dans le README) ;
  avec --ajouter, seuls les audios absents des résultats sont mesurés
  (pour compléter le jeu plus tard sans remesurer les anciens) ;
- rien n'est réglé ensuite sur ce jeu (seuil, prompt, FAQ).

Issue de chaque question :
- bonne  : la bonne question de la FAQ, ou un refus pour une question hors FAQ ;
- refus  : « je ne sais pas » alors que la question est dans la FAQ ;
- fausse : une mauvaise question de la FAQ, ou une réponse à une question hors FAQ.

Le TEM (taux d'erreur sur les mots de l'ASR) n'est calculé que pour les
lignes dont la colonne transcription_reference est remplie.

Résultats par question : results/resultats_test.csv (transcriptions et
locuteurs anonymes, aucun audio).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from scripts.evaluer_dev_audio import erreurs_mots
from src.config import AUDIO_TEST_DIR, RESULTATS_DIR

MANIFESTE_TEST = AUDIO_TEST_DIR / "manifeste.csv"
RESULTATS_TEST = RESULTATS_DIR / "resultats_test.csv"
COLONNES_REQUISES = ("fichier", "id_question_attendue", "locuteur")
ETAPES = ("asr_s", "recherche_s", "tts_s", "total_s")


def charger_manifeste(chemin: Path = MANIFESTE_TEST) -> list[dict]:
    """Lit et vérifie le manifeste ; `attendu` vaut None pour une question hors FAQ."""
    with open(chemin, encoding="utf-8", newline="") as fichier:
        lignes = list(csv.DictReader(fichier))
    manquantes = [c for c in COLONNES_REQUISES if lignes and c not in lignes[0]]
    if manquantes:
        raise ValueError(f"Colonnes absentes du manifeste : {manquantes}")
    noms = [lg["fichier"] for lg in lignes]
    doublons = sorted({n for n in noms if noms.count(n) > 1})
    if doublons:
        raise ValueError(f"Fichiers en double dans le manifeste : {doublons}")
    for lg in lignes:
        brut = lg["id_question_attendue"].strip()
        lg["attendu"] = int(brut) if brut else None
        if lg["attendu"] is not None and not 1 <= lg["attendu"] <= 30:
            raise ValueError(f"{lg['fichier']} : id_question_attendue hors de 1..30 ({brut})")
    return lignes


def _ou_rien(valeur, conversion, defaut=None):
    """Convertit une cellule (texte du CSV ou valeur Python) ; vide ou None donne `defaut`."""
    if valeur is None or str(valeur).strip() in ("", "None"):
        return defaut
    return conversion(valeur)


def issue(attendu: int | None, obtenu: int | None) -> str:
    """Classe une réponse en bonne / refus / fausse (voir la docstring du module)."""
    if obtenu == attendu:
        return "bonne"
    if obtenu is None:
        return "refus"
    return "fausse"


def _quantile(valeurs: list[float], q: float) -> float:
    """Quantile par interpolation linéaire (même convention que numpy et pandas)."""
    valeurs = sorted(valeurs)
    position = q * (len(valeurs) - 1)
    bas = int(position)
    haut = min(bas + 1, len(valeurs) - 1)
    return valeurs[bas] + (valeurs[haut] - valeurs[bas]) * (position - bas)


def resumer(lignes: list[dict]) -> dict:
    """Calcule les métriques globales à partir des lignes de résultats."""
    n = len(lignes)
    issues = {i: sum(lg["issue"] == i for lg in lignes) for i in ("bonne", "refus", "fausse")}
    dans_faq = [lg for lg in lignes if lg["attendu"] is not None]
    hors_faq = [lg for lg in lignes if lg["attendu"] is None]
    avec_ref = [lg for lg in lignes if lg.get("erreurs_mots") not in (None, "")]
    mots = sum(lg["mots_reference"] for lg in avec_ref)
    return {
        "n": n,
        **issues,
        "n_dans_faq": len(dans_faq),
        "bonnes_dans_faq": sum(lg["issue"] == "bonne" for lg in dans_faq),
        "n_hors_faq": len(hors_faq),
        "refus_hors_faq": sum(lg["issue"] == "bonne" for lg in hors_faq),
        "tem": sum(lg["erreurs_mots"] for lg in avec_ref) / mots if mots else None,
        "n_tem": len(avec_ref),
        "latences": {
            e: (statistics.median(v), _quantile(v, 0.95))
            for e in ETAPES
            if (v := [lg[e] for lg in lignes])
        },
    }


def _afficher(resume: dict, lignes: list[dict], sur_gpu: bool) -> None:
    n = resume["n"]
    print(f"\n{n} questions, {len({lg['locuteur'] for lg in lignes})} locuteurs")
    for i in ("bonne", "refus", "fausse"):
        print(f"  {i:7} {resume[i]:3}  ({resume[i] / n:.0%})")
    if resume["n_dans_faq"]:
        print(f"  dans la FAQ : {resume['bonnes_dans_faq']}/{resume['n_dans_faq']} bonnes")
    if resume["n_hors_faq"]:
        print(f"  hors FAQ    : {resume['refus_hors_faq']}/{resume['n_hors_faq']} refusées")
    if resume["tem"] is not None:
        print(f"  TEM de l'ASR : {resume['tem']:.1%} (sur {resume['n_tem']} transcriptions)")

    print("\nPar locuteur :")
    for loc in sorted({lg["locuteur"] for lg in lignes}):
        siennes = [lg for lg in lignes if lg["locuteur"] == loc]
        bonnes = sum(lg["issue"] == "bonne" for lg in siennes)
        print(f"  {loc} : {bonnes}/{len(siennes)}")

    print("\nLatences (médiane / p95) :")
    for etape, (mediane, p95) in resume["latences"].items():
        print(f"  {etape:12} {mediane:6.3f} s / {p95:6.3f} s")
    if not sur_gpu:
        print("  ATTENTION : mesure sur CPU, latences de l'ASR non représentatives.")

    echecs = [lg for lg in lignes if lg["issue"] != "bonne"]
    if echecs:
        print("\nÉchecs :")
        for lg in echecs:
            print(
                f"  {lg['fichier']:22} attendu {lg['attendu']!s:4} obtenu "
                f"{lg['id_question']!s:4} (1re candidate {lg['id_top1']}, "
                f"score {lg['score'] or 0:.3f}) « {lg['transcription']} »"
            )


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--dossier", type=Path, default=AUDIO_TEST_DIR, help="dossier des audios")
    parseur.add_argument("--manifeste", type=Path, default=MANIFESTE_TEST)
    parseur.add_argument("--sortie", type=Path, default=RESULTATS_TEST)
    groupe = parseur.add_mutually_exclusive_group()
    groupe.add_argument("--refaire", action="store_true", help="écrase les résultats existants")
    groupe.add_argument("--ajouter", action="store_true", help="mesure les nouveaux audios")
    args = parseur.parse_args()

    manifeste = charger_manifeste(args.manifeste)
    deja = []
    if args.sortie.exists():
        if not (args.refaire or args.ajouter):
            raise SystemExit(
                f"{args.sortie} existe déjà : chaque audio ne se mesure qu'une fois. "
                "--ajouter pour mesurer les nouveaux audios, --refaire pour tout écraser."
            )
        if args.ajouter:
            with open(args.sortie, encoding="utf-8", newline="") as fichier:
                deja = list(csv.DictReader(fichier))
    mesures = {lg["fichier"] for lg in deja}
    a_mesurer = [m for m in manifeste if m["fichier"] not in mesures]

    absents = [m["fichier"] for m in a_mesurer if not (args.dossier / m["fichier"]).is_file()]
    if absents:
        raise SystemExit(f"{len(absents)} audio(s) introuvable(s) dans {args.dossier} : {absents}")
    if not a_mesurer:
        raise SystemExit("Rien à mesurer : tous les audios du manifeste ont déjà un résultat.")

    import ctranslate2  # import tardif : les vérifications ci-dessus restent instantanées

    from src.pipeline import prechauffer, repondre
    from src.search import chercher

    sur_gpu = ctranslate2.get_cuda_device_count() > 0
    print(f"Préchauffage ({'GPU' if sur_gpu else 'CPU'})…")
    prechauffer()

    nouvelles = []
    for n, m in enumerate(a_mesurer, 1):
        r = repondre(args.dossier / m["fichier"])
        reference = (m.get("transcription_reference") or "").strip()
        erreurs, nb_mots = erreurs_mots(reference, r.transcription) if reference else ("", "")
        ligne = {k: v for k, v in m.items() if k != "attendu"}
        ligne.update(r.vers_ligne())
        del ligne["audio_reponse"], ligne["simulation"]
        ligne.update(
            {
                # 1re candidate avant le seuil : dit si un refus aurait été juste ou faux
                "id_top1": chercher(r.transcription).entree.id if r.transcription else "",
                "erreurs_mots": erreurs,
                "mots_reference": nb_mots,
                "issue": issue(m["attendu"], r.id_question),
                "materiel": "gpu" if sur_gpu else "cpu",
            }
        )
        nouvelles.append(ligne)
        print(f"\r{n}/{len(a_mesurer)}", end="", flush=True)

    toutes = [*deja, *nouvelles]
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    colonnes = list(dict.fromkeys(c for lg in toutes for c in lg))
    with open(args.sortie, "w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=colonnes, lineterminator="\n")
        ecrivain.writeheader()
        ecrivain.writerows(toutes)

    # Les lignes relues du CSV sont du texte : on les remet au bon type pour le résumé.
    for lg in toutes:
        lg["attendu"] = _ou_rien(lg["id_question_attendue"], int)
        lg["id_question"] = _ou_rien(lg["id_question"], int)
        lg["score"] = _ou_rien(lg["score"], float)
        for e in ETAPES:
            lg[e] = float(lg[e])
        for c in ("erreurs_mots", "mots_reference"):
            lg[c] = _ou_rien(lg[c], int, defaut="")
    _afficher(resumer(toutes), toutes, sur_gpu)
    print(f"\nRésultats : {args.sortie}")


if __name__ == "__main__":
    main()
