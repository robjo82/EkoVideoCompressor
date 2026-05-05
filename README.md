# EkoVideo Compressor

Application desktop macOS pour compresser rapidement des enregistrements de réunions visio.

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

Le workflow GitHub Actions `.github/workflows/release-macos.yml` publie automatiquement une release macOS quand un tag `vX.Y.Z` est poussé.

Exemple:

```bash
git tag v0.1.0
git push origin v0.1.0
```

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

## Détection des locuteurs (diarisation)

Pour identifier qui parle quand dans une réunion à plusieurs voix, l'app utilise [pyannote.audio 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). Setup en une fois par poste :

1. Créez un compte gratuit sur huggingface.co (ou connectez-vous).
2. Acceptez les licences (un clic chacune) :
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
3. Générez un token Read sur https://huggingface.co/settings/tokens
4. Ouvrez `Réglages` → onglet `Transcription` → cochez **Détection des locuteurs** et collez le token dans **Token Hugging Face**.
5. Le bouton `Installer MLX Whisper` (onglet `Transcrire`) installe désormais aussi pyannote + torch (~2 Go, 5-10 min la première fois).

La transcription produit alors des segments préfixés `[SPEAKER_00]`, `[SPEAKER_01]`, etc. Vous pouvez renommer les locuteurs dans le fichier de sortie. Formats supportés : txt, srt, vtt, json, tsv.

Sans token ou pyannote installé, l'app retombe sur la transcription standard (sans étiquettes locuteur) avec un avertissement.

## Builds GitHub

- Un push sur `main` lance un build macOS de test disponible dans les artifacts GitHub Actions.
- Une release installable par l'auto-update est publiée uniquement quand un tag `vX.Y.Z` est poussé.

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
