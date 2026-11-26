# Données

| Fichier | Contenu | Versionné |
|---|---|---|
| `faq.csv` | Les 30 questions-réponses de la FAQ Western Union Sénégal | oui |
| `dev/questions_dev.csv` | Jeu de développement : 90 reformulations orales (3 par question) et 15 questions hors FAQ | oui |
| `dev_audio/` | Les mêmes questions lues par 8 voix synthétiques (`python -m scripts.generer_dev_audio`) | non, régénérable |
| `audio_test/` | Jeu de test oral de B (voix réelles), voir `audio_test/LISEZMOI.md` | manifeste seulement |

**Règle d'or :** on règle le système (modèle, seuil, `question_recherche`) sur les
jeux de **dev**, jamais sur `audio_test/`. Sinon le score final est truqué.

## Colonnes de `faq.csv`

| Colonne | Rôle |
|---|---|
| `question` | Question officielle, affichée à l'utilisateur |
| `reponse` | Réponse écrite, affichée |
| `reponse_vocale` | Même réponse, rédigée pour être lue à voix haute (sigles épelés, « francs CFA ») |
| `question_recherche` | Texte indexé par la recherche (avec la réponse), jamais affiché |
| `url_source` | Page officielle d'où vient la réponse |

### Comment écrire `question_recherche`

Elle reformule la question **comme un client la poserait**, en mettant en avant
ce qui la **distingue de ses voisines**. Exemple : la n°30 (documents pour le
retrait) se faisait capter par la n°23 (comment retirer), parce que les deux
parlent de « retirer ». Sa version de recherche insiste donc sur « documents,
pièces d'identité, passeport ».

Règles :

1. **Ne jamais recopier une phrase du jeu de dev** : on testerait le système
   sur ses propres réponses. On ajoute des synonymes courants (« preuve »,
   « liquide », « justificatif »), pas des formulations entières.
2. **Mesurer chaque modification** sur les deux jeux de dev (texte et audio)
   et **chercher les régressions**. Une reformulation trop large devient un
   « attracteur » qui capte les questions des autres.
3. **Ne pas retoucher indéfiniment** : chaque retouche faite en regardant le
   dev rend les chiffres du dev plus optimistes. Seul le test de B tranche.

Mesures de la passe actuelle (17 questions réécrites) : top1 sur le dev texte
73,3 % -> 82,2 %, sur le dev audio 64,4 % -> 76,7 %.
