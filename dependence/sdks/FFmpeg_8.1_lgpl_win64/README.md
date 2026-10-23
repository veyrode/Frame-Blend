# FFmpeg 8.1 (LGPL, shared) for the Premiere file reader

`plugins/adobe/fb_media_reader.cpp` decodes recordings with FFmpeg, because Premiere hands effects only frames on the sequence's frame grid.

- Source: https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip
- Downloaded 2026-10-05, 80,984,510 bytes, SHA-256 `ed72154c220438057e840421c8df46e731c707aeecabec25ada7b27d9034f72e`
- License: LGPL 3 (`LICENSE.txt`); the plugin links the shared libraries dynamically.

Tracked here: `include/`, the `.def` files and MSVC import libraries regenerated from them (`lib.exe /def:avformat-62.def /out:avformat.lib /machine:x64`, likewise for avcodec and avutil): the archive's MinGW import libraries cannot be delay-loaded by MSVC (LNK4199). Not tracked (too large for Git): `bin/avutil-60.dll`, `bin/swresample-6.dll`, `bin/avcodec-62.dll`, `bin/avformat-62.dll`. Restore them from the archive's `bin/` folder; `tools/install_plugins.ps1` copies them next to `FrameBlend.aex`.
