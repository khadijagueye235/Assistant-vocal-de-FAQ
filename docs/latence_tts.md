# Latence de la synthèse vocale : phrase par phrase ou cache ?

Le sujet propose une extension : **synthétiser la réponse phrase par phrase**, pour
que l'utilisateur entende la première phrase pendant que les suivantes sont en
cours de synthèse. Avant de l'implémenter, on a mesuré.

## Constat de départ

Après l'ajout de la voix, le TTS était l'étape la plus lente du pipeline sur
Colab (environ 55 % du temps total) :

| Étape | Colab (GPU T4) |
|---|---|
| ASR | 0,33 à 0,42 s |
| Recherche | 14 à 21 ms |
| **TTS** | **0,44 à 0,66 s** |
| Total | 0,85 à 1,01 s |

## Mesure des options (PC local, médiane de 3 essais, voix Denise)

| Réponse | Synthèse complète | 1er morceau du flux Edge | 1er groupe de mots seul |
|---|---|---|---|
| n°14 (139 car.) | 1,65 s | 1,04 s | 1,34 s (« Il faut avoir 18 ans, ») |
| n°23 (139 car.) | 1,64 s | 1,04 s | 1,33 s (« En agence, ») |
| n°5 (125 car.) | 1,45 s | 1,03 s | 1,33 s (« Avec un profil vérifié : ») |
| n°30 (176 car.) | 1,65 s | 1,03 s | 1,44 s (« Résidents du Sénégal : ») |

Deux constats :

1. **Chaque requête à Edge TTS coûte environ 1 s fixe** (connexion au service,
   échanges de sécurité), quelle que soit la longueur du texte : deux mots
   prennent 1,33 s, 140 caractères 1,65 s. Découper la réponse multiplie ce coût
   fixe : la première phrase arriverait **plus tard** qu'en lisant le flux
   d'une seule requête.
2. **24 réponses sur 30 ne font qu'une phrase** (`reponse_vocale`, environ
   120 caractères en moyenne) : un découpage par phrase n'y changerait rien.

**Conclusion : la synthèse phrase par phrase n'apporte pas de gain ici.**

## La solution retenue : un cache audio

Les réponses de la FAQ sont **fixes** : il n'y a que 32 messages possibles
(30 réponses, « je ne sais pas », « rien entendu »). Chacun est synthétisé une
seule fois, puis réutilisé :

- le fichier est nommé d'après une empreinte de la voix, du débit et du texte :
  changer de voix ou modifier une réponse produit automatiquement un nouveau
  fichier ;
- `prechauffer()` remplit le cache au démarrage, 8 requêtes à la fois ;
- sans réseau au démarrage, l'assistant démarre quand même et synthétise à la
  demande ;
- `TTS_CACHE=0` désactive le cache, pour mesurer la latence réelle d'Edge TTS.

| | Sans cache | **Avec cache** |
|---|---|---|
| TTS par question (PC local) | environ 1,6 s | **0,5 ms** |
| Total par question (PC local) | environ 3,6 s | **environ 2,0 s** (presque tout pour l'ASR) |
| Démarrage | — | + 8 s, une seule fois (32 voix) |

## Quand le flux redeviendrait utile

Si la réponse n'était plus fixe (par exemple reformulée par un LLM), le cache
ne servirait plus. Il faudrait alors lire le **flux** d'Edge TTS
(`Communicate.stream`) : le premier son arrive après environ 1 s au lieu de
1,6 s, et non découper le texte en phrases.
