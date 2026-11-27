# Jeu de test oral

30 questions posées à voix haute par des personnes **extérieures au groupe**,
avec leur accord. Ce jeu sert **uniquement à mesurer**, jamais à régler
le système (ni le seuil, ni le prompt, ni le choix des modèles).

## Où sont les audios ?

Les enregistrements ne sont **pas** dans le dépôt : ce sont des voix de
personnes réelles, donc des données personnelles. Ils sont stockés sur le
Drive du groupe et copiés dans ce dossier avant de lancer les mesures.
Seul `manifeste.csv` est versionné.

## Colonnes de `manifeste.csv`

| Colonne | Contenu | Exemple |
|---|---|---|
| `fichier` | Nom du fichier audio dans ce dossier | `q07_L05.m4a.mp4` |
| `id_question_attendue` | `id` de la question de `data/faq.csv` qui doit être retrouvée, vide si la question est hors FAQ | `7` |
| `locuteur` | Identifiant anonyme, jamais le nom | `L03` |
| `consentement` | `oui` uniquement si l'accord a été recueilli | `oui` |
| `date_enregistrement` | Format AAAA-MM-JJ | `2026-11-28` |
| `transcription_reference` | Ce qui est **réellement dit**, écrit à l'écoute avant toute mesure (sert à calculer le taux d'erreur de l'ASR) | `euh comment je sais si mon argent est arrivé` |
| `particularites` | Ce qui rend la question difficile, séparé par `;` | `hésitation;wolof;bruit` |

## Consignes d'enregistrement

- Formuler la question **comme on la dirait** au guichet, pas en lisant la FAQ :
  hésitations, reformulations et mots en wolof sont bienvenus.
- Un fichier par question, 2 à 15 secondes.
- Varier les locuteurs (au moins 5 personnes différentes).
