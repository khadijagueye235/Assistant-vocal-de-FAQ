# Avancement du projet

*Mettre à jour une ligne à chaque brique terminée.*

| # | Brique | Responsable | État | Fichier |
|---|---|---|---|---|
| 1 | Choix du service et FAQ (30 questions) | B | ✅ fait | `data/faq.csv` |
| 2 | Nettoyage de la FAQ | A | ✅ fait | `data/faq.csv` |
| 3 | ASR (faster-whisper + VAD) | A | ✅ testé (audio synthétique, CPU local et GPU Colab) | `src/asr.py` |
| 4 | Recherche par embeddings | A | ⬜ à faire | `src/search.py` |
| 5 | TTS (Edge TTS) | A | ⬜ à faire | `src/tts.py` |
| 6 | Assemblage `repondre()` | A | 🟡 contrat figé, recherche et TTS simulés | `src/pipeline.py`, `docs/contrat_repondre.md` |
| 7 | Interface Gradio | C | ⬜ à faire | `app/` |
| 8 | Jeu de test oral (30 audios) | B | ⬜ à faire (manifeste prêt) | `data/audio_test/manifeste.csv` |
| 9 | Mesures (taux, latences) | B | ⬜ à faire | `results/` |
| 10 | Analyse des échecs | A + B | ⬜ à faire | `results/cas_d_echec.csv` |
| 11 | README, fiche technique, support | C | ⬜ à faire | `README.md`, `docs/` |

**Légende :** ✅ fait · 🟡 en cours · ⬜ à faire
