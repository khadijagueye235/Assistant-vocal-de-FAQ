# Contrat de `repondre(audio)`

Point d'entrée unique du pipeline, utilisé par l'interface Gradio (C) et par
les mesures (B). La signature et les champs ci-dessous ne changeront plus :
on peut construire dessus dès maintenant.

```python
from src.asr import prechauffer
from src.pipeline import repondre

prechauffer()                 # une fois au démarrage
r = repondre("question.m4a")  # ou le tuple / le chemin fourni par gr.Audio
```

## Entrée

`audio` accepte :

- un **chemin de fichier** (wav, mp3, m4a, ogg…) ;
- le **tuple `(fréquence, tableau)`** renvoyé par `gr.Audio(type="numpy")` ;
- un **tableau numpy** mono float32 déjà à 16 kHz.

## Sortie : objet `Reponse`

| Champ | Type | Contenu |
|---|---|---|
| `transcription` | `str` | Ce que l'ASR a compris (vide si aucune parole) |
| `id_question` | `int` ou `None` | `id` de la question retenue dans `data/faq.csv` ; `None` si aucune |
| `question_trouvee` | `str` ou `None` | Texte de cette question |
| `score` | `float` ou `None` | Similarité de la recherche (plus c'est haut, plus c'est proche) |
| `reponse_texte` | `str` | Réponse à afficher, identique à celle qui est lue |
| `audio_reponse` | `Path` ou `None` | Fichier audio de la réponse parlée (voix synthétique) |
| `latences.asr_s` | `float` | Durée de la transcription, en secondes |
| `latences.recherche_s` | `float` | Durée de la recherche, en secondes |
| `latences.tts_s` | `float` | Durée de la synthèse vocale, en secondes |
| `latences.total_s` | `float` | Durée de bout en bout, en secondes |
| `simulation` | `bool` | `True` tant qu'une étape est encore simulée |
| `a_trouve` | `bool` (propriété) | `True` si une question a été retenue |
| `vers_ligne()` | méthode | Tout à plat dans un `dict`, prêt pour un CSV |

Les latences sont **brutes** : arrondir uniquement à l'affichage.

## Cas particuliers

- **Aucune parole détectée** : `transcription` vide, `id_question` à `None`,
  `reponse_texte` = « Je n'ai rien entendu… », qui est aussi lu à voix haute.
- **Question hors FAQ** (à venir avec le seuil) : `id_question` à `None`,
  `reponse_texte` = message « je ne sais pas ».

## État actuel

| Étape | État |
|---|---|
| ASR | Réel (faster-whisper + VAD) |
| Recherche | **Simulée** : ne retient aucune question |
| TTS | **Simulé** : 0,5 s de silence |

## Exemple pour l'interface (C)

```python
import gradio as gr
from src.asr import prechauffer
from src.pipeline import repondre

prechauffer()

def traiter(audio):
    r = repondre(audio)
    l = r.latences
    latences = (f"ASR {l.asr_s:.2f} s · recherche {l.recherche_s * 1000:.0f} ms · "
                f"TTS {l.tts_s:.2f} s · total {l.total_s:.2f} s")
    return r.transcription, r.question_trouvee or "—", r.reponse_texte, r.audio_reponse, latences

gr.Interface(
    traiter,
    gr.Audio(sources=["microphone"], type="filepath"),
    [gr.Textbox(label="Transcription"), gr.Textbox(label="Question retrouvée"),
     gr.Textbox(label="Réponse"), gr.Audio(label="Réponse parlée (voix synthétique)"),
     gr.Textbox(label="Latences")],
).launch(share=True)
```

## Exemple pour les mesures (B)

```python
import pandas as pd
from src.asr import prechauffer
from src.config import AUDIO_TEST_DIR, RESULTATS_DIR
from src.pipeline import repondre

prechauffer()
manifeste = pd.read_csv(AUDIO_TEST_DIR / "manifeste.csv")
lignes = []
for _, m in manifeste.iterrows():
    r = repondre(AUDIO_TEST_DIR / m.fichier)
    lignes.append({**m.to_dict(), **r.vers_ligne(),
                   "correct": r.id_question == m.id_question_attendue})
pd.DataFrame(lignes).to_csv(RESULTATS_DIR / "resultats.csv", index=False)
```
