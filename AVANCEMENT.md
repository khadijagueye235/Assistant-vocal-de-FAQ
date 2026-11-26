# Avancement du projet

*Mettre à jour une ligne à chaque brique terminée.*

| # | Brique | Responsable | État | Fichier |
|---|---|---|---|---|
| 1 | Choix du service et FAQ (30 questions) | B | ✅ fait | `data/faq.csv` |
| 2 | Nettoyage de la FAQ | A | ✅ fait | `data/faq.csv` |
| 3 | ASR (faster-whisper + VAD) | A | ✅ vocabulaire métier dans le prompt ; TEM 11 % sur le dev audio (wolof 37 %) | `src/asr.py` |
| 4 | Recherche par embeddings | A | ✅ e5-small, index Q+R, requêtes normalisées, question_recherche réécrite pour 17 questions ; top1 dev texte 82 %, dev audio 77 % | `src/search.py`, `scripts/` |
| 5 | TTS (Edge TTS) | A | ✅ voix Denise, cache audio (TTS ~0,5 ms), sans voix si le réseau tombe | `src/tts.py` |
| 6 | Assemblage `repondre()` | A | ✅ pipeline complet, plus rien de simulé | `src/pipeline.py`, `docs/contrat_repondre.md` |
| 7 | Interface Gradio | C | ⬜ à faire | `app/` |
| 8 | Jeu de test oral (30 audios) | B | ⬜ à faire (manifeste prêt) | `data/audio_test/manifeste.csv` |
| 9 | Mesures (taux, latences) | B | ⬜ à faire | `results/` |
| 10 | Analyse des échecs | A + B | ⬜ à faire | `results/cas_d_echec.csv` |
| 11 | README, fiche technique, support | C | 🟡 README de base, licence, CI et notebook Colab posés par A ; à compléter (interface, résultats du test, noms) | `README.md`, `docs/` |

**Légende :** ✅ fait · 🟡 en cours · ⬜ à faire
