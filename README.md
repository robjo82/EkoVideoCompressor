# EkoVideo Compressor

Application desktop macOS pour compresser rapidement des enregistrements de réunions visio.

## Installation équipe Mac (Apple Silicon)

1. Ouvrez la page **Releases** du dépôt GitHub.
2. Téléchargez l'archive `EkoVideoCompressor-macos-arm64-vX.Y.Z.zip`.
3. Décompressez et déplacez `EkoVideoCompressor.app` dans `/Applications`.
4. Premier lancement: clic droit sur l'app, puis `Open`.

Si macOS bloque l'app ou ffmpeg, exécutez:

```bash
xattr -dr com.apple.quarantine /Applications/EkoVideoCompressor.app
```

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
