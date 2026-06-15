# EkoVideo Compressor

Application desktop macOS pour compresser rapidement des enregistrements de
réunions visio. Accepte aussi les **enregistrements audio seuls** (mp3,
m4a, wav, flac, ogg, opus…) — la sortie est alors un `.m4a` (AAC) au
lieu d'un `.mp4` H.265.

## Installation équipe Mac (Apple Silicon)

L'app est signée ad-hoc (pas de compte développeur Apple payant). Au **premier
lancement seulement**, macOS demande une confirmation explicite. Les mises à
jour automatiques sont ensuite transparentes.

1. Ouvrez la page **Releases** du dépôt GitHub.
2. Téléchargez `EkoVideoCompressor-macos-arm64-vX.Y.Z.zip`.
3. Décompressez et déplacez `EkoVideoCompressor.app` dans `/Applications`.
4. **Premier lancement** : clic droit sur l'app → `Ouvrir` → `Ouvrir` dans
   la boîte de dialogue. Les lancements suivants se font normalement.

Si macOS affiche `app endommagée` ou bloque ffmpeg, le bundle a été mis en
quarantaine par le navigateur. Une seule commande suffit :

```bash
xattr -dr com.apple.quarantine /Applications/EkoVideoCompressor.app
```

Puis double-cliquez normalement.

## Release automatisée

Le workflow GitHub Actions `.github/workflows/release-macos.yml` utilise semantic-release.
Un push sur `main` déclenche les tests, calcule la prochaine version à partir des messages de commit conventionnels, crée le tag GitHub, build l'app macOS et publie la release installable par l'auto-update.

## Auto-update in-app

Un bouton `Vérifier mise à jour` est disponible dans le header de l'application.

Flux:
1. L'app vérifie la dernière release GitHub.
2. Si une version plus récente existe, elle télécharge l'archive macOS arm64.
3. L'app installe la nouvelle version et se relance.

Si le dépôt est privé, renseignez un token GitHub (lecture du repo) dans:
`Paramètres` → `Token update`.

## Transcription locale

L'app peut lancer une transcription locale via **MLX Whisper** sur Apple Silicon.
La vidéo originale est utilisée comme source audio, puis l'audio temporaire est extrait en WAV propre avant transcription.

Dans l'onglet `Transcrire`, le bouton `Installer MLX Whisper` crée un environnement isolé dans
`~/Library/Application Support/EkoVideo Compressor/mlx-whisper-venv`.
L'app ne modifie pas le Python système/Homebrew.

L'installation automatique requiert Python 3.11, 3.12 ou 3.13 disponible sur le Mac.
Si besoin:

```bash
brew install python@3.12
```

Dans l'onglet `Transcrire`, le modèle recommandé par défaut est:
`mlx-community/whisper-large-v3-turbo`.

Le champ `Contexte` sert à ajouter les noms propres, clients, projets, acronymes et termes métier qui doivent guider Whisper. Le contenu est automatiquement formaté pour Whisper (phrase d'amorce en français — le modèle le traite comme du vocabulaire attendu, pas comme une liste).

## Transcription Cloud (API)

En complément du moteur local, un mode **Cloud** envoie l'audio à une API
distante, souvent meilleure que la chaîne locale sur les audios
difficiles. Deux familles de modèles sont prises en charge derrière une
interface unique :

- **LLM multimodaux** (Gemini) : un seul appel renvoie la transcription
  horodatée, la détection des locuteurs, un titre et les termes
  techniques.
- **STT dédiés** (AssemblyAI, OpenAI gpt-4o-transcribe, Gladia,
  Deepgram) : transcription + diarisation native, souvent à plus bas
  coût ; le titre, les noms d'interlocuteurs et les corrections métier
  sont ajoutés par la passe LLM locale quand elle est installée.

Fournisseurs et tarifs indicatifs (par heure d'audio) :

| Fournisseur / modèle | ≈ coût/h | Particularité |
|---|---|---|
| Gemini 3.5 Flash (défaut) | ~0,50 $ | bundle complet (titre + noms) |
| Gemini 3.1 Flash-Lite | ~0,11 $ | le moins cher des LLM |
| AssemblyAI Universal-3 | ~0,21 $ | WER FR de pointe |
| OpenAI gpt-4o-transcribe (diarisation) | ~0,36 $ | pilotable par prompt |
| OpenAI gpt-4o-mini-transcribe | ~0,18 $ | sans diarisation |
| Gladia Solaria-3 | ~0,20–0,61 $ | **UE / RGPD**, top WER EU |
| Gladia Solaria-1 | ~0,20–0,61 $ | UE / RGPD, 100+ langues |
| Deepgram Nova-3 | ~0,26 $ | rapide |

- Choix du moteur (Local / Cloud) et du modèle au lancement de chaque
  file, avec le **coût estimé affiché avant l'envoi**.
- **Une clé API par fournisseur** et un **budget mensuel plafond** dans
  `Réglages` → `Transcription Cloud`. Le moteur refuse tout traitement
  dont l'estimation dépasse le budget restant ; la consommation réelle
  (tokens/durée + coût) est suivie par traitement et par mois.
- Les modèles distants sont listés dans l'onglet `Modèles` avec leur
  prix (au token ou à l'heure selon le fournisseur).
- L'historique (`Bibliothèque`) indique le modèle utilisé pour chaque
  transcription via la colonne `Modèle` — pratique pour comparer les
  moteurs entre eux.
- Le contexte déjà connu de l'app est transmis aux API : vocabulaire
  métier + noms attendus (boosting de termes natif de chaque
  fournisseur) et nombre d'intervenants attendu (config de diarisation),
  pour améliorer la qualité sans surcoût.
- L'audio est compressé (MP3 mono 16 kHz), envoyé par fenêtres de
  30 minutes puis **supprimé des serveurs du fournisseur** sitôt la
  réponse reçue.
- En cas d'erreur réseau ou de quota, le traitement bascule
  automatiquement sur le moteur local avec un avertissement ; une clé
  refusée ou un budget atteint échoue explicitement.

## Amélioration locale progressive

Après la transcription Whisper, l'app lance une passe locale via MLX-LM si nécessaire.
Elle installe `mlx-lm` automatiquement dans le même environnement isolé que MLX Whisper, sans commande Terminal côté utilisateur.

Cette passe utilise par défaut `mlx-community/Mistral-7B-Instruct-v0.3-4bit` pour :

- proposer un titre ;
- associer les locuteurs à des noms quand le dialogue le permet ;
- appliquer uniquement des corrections textuelles à forte confiance ;
- lister les passages douteux dans un fichier `- à vérifier.md` ;
- relancer Whisper sur quelques extraits audio ciblés autour des timestamps douteux.

Le fichier brut Whisper reste disponible. Si des améliorations fiables existent, l'app écrit un second fichier `- améliorée`.

## Architecture moteur / SwiftUI

Le moteur métier est désormais exposé par un package Python headless
`ekovideo_engine`. Il parle en JSONL pour permettre à l'interface SwiftUI
native de piloter les traitements sans dépendre de PySide.

Commandes utiles :

```bash
python -m ekovideo_engine --smoke-test
python -m ekovideo_engine model-list
python -m ekovideo_engine library-list
python -m transcription_eval.evaluate --min-score 0.95
```

Pendant la migration, l'interface PySide reste disponible et le build macOS
embarque aussi l'exécutable `Contents/Resources/engine/ekovideo-engine`.
La nouvelle base SwiftUI est dans `macos/EkoVideoCompressor`.

## Détection des locuteurs (diarisation)

Pour identifier qui parle quand dans une réunion à plusieurs voix, l'app utilise pyannote.audio. Setup en une fois par poste :

1. Créez un compte gratuit sur huggingface.co (ou connectez-vous).
2. Acceptez les licences (un clic chacune) :
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. Générez un token Read sur https://huggingface.co/settings/tokens
4. Ouvrez `Réglages` → onglet `Transcription` → cochez **Détection des locuteurs** et collez le token dans **Token Hugging Face**.
5. Le bouton `Installer MLX Whisper` (onglet `Transcrire`) installe désormais aussi pyannote + torch (~2 Go, 5-10 min la première fois).

La transcription produit alors des segments préfixés `[SPEAKER_00]`, `[SPEAKER_01]`, etc. Vous pouvez renommer les locuteurs dans le fichier de sortie. Formats supportés : txt, srt, vtt, json, tsv.

Sans token ou pyannote installé, l'app retombe sur la transcription standard (sans étiquettes locuteur) avec un avertissement.

## Builds GitHub

- Un push sur `main` lance un build macOS de test disponible dans les artifacts GitHub Actions.
- Si semantic-release détecte un changement publiable, une release installable par l'auto-update est publiée automatiquement.

## Build local macOS

```bash
scripts/build_macos.sh 0.1.0
```

Le zip final est produit dans `dist/release/`.

## Secrets préparés pour signature/notarization future

- `APPLE_CERTIFICATE_BASE64`
- `APPLE_CERT_PASSWORD`
- `APPLE_ID`
- `APPLE_APP_PASSWORD`
- `TEAM_ID`
