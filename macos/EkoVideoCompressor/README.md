# EkoVideoCompressor SwiftUI shell

This is the native macOS frontend for the headless Python engine.

Development run:

```bash
cd macos/EkoVideoCompressor
EKOVIDEO_ENGINE=/usr/bin/python3 swift run EkoVideoCompressor
```

When the engine is a Python interpreter, the app automatically calls
`python -m ekovideo_engine ...`. Packaged releases will instead place a
standalone `ekovideo-engine` executable in `Contents/Resources/engine/`.
