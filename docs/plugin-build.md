# Building Frame Blend (Windows)

## What you need

- **Rust** with the MSVC target (`rustup default stable-x86_64-pc-windows-msvc`).
- **Visual Studio 2022 or later Build Tools** (C++ workload: MSVC, Windows SDK, CMake).
- **CUDA toolkit** (build time only): `nvcc` compiles `plugins/common/gpu/fb_blend.cu` to PTX. At run time the plugin loads `nvcuda.dll` and `OpenCL.dll` dynamically, so it carries no CUDA runtime and still loads on machines that have only one of them.
- **Adobe SDKs** (free, Adobe ID needed; not in this repository because Adobe's licence does not allow redistributing them), from [developer.adobe.com/console/servicesandapis](https://developer.adobe.com/console/servicesandapis):
  - the **After Effects SDK** (Windows). `-AeSdk` / `AE_SDK_PATH` is its `Examples` folder (it holds `Headers\AE_Effect.h` and `Resources\PiPLtool.exe`);
  - the **Premiere Pro C++ SDK** (Windows). `-PrSdk` / `PR_SDK_PATH` is the folder holding `Headers\PrSDKGPUFilter.h` (in the SDK zip, the `Examples` folder). Version 24.0 or later.
- **FFmpeg 8.1 shared DLLs** (LGPL): headers and import libraries are in `dependence/sdks/FFmpeg_8.1_lgpl_win64`; the four DLLs come from the archive named in [its README](../dependence/sdks/FFmpeg_8.1_lgpl_win64/README.md) and go in its `bin` folder.
- For the installer: **Inno Setup 6** (`winget install JRSoftware.InnoSetup`) and Adobe's **ZXPSignCmd** ([Adobe-CEP/CEP-Resources](https://github.com/Adobe-CEP/CEP-Resources), folder `ZXPSignCMD`), which signs the Premiere companion.

By default the scripts look for the SDKs under `dependence/sdks/Adobe` and ZXPSignCmd under `dependence/sdks/ZXPSignCmd_4.1.103_win64` (both ignored by Git); pass other locations as parameters or set `FRAMEBLEND_AE_SDK` and `FRAMEBLEND_PR_SDK`.

## Installer (the easy way)

```powershell
.\tools\build_installer.ps1 -AeSdk 'C:\SDK\AfterEffectsSDK\Examples' -PrSdk 'C:\SDK\PremiereProSDK\Examples'
```

It builds the core and the plugin, signs the Premiere companion if its sources changed (`tools/sign_panel.ps1 -Tool <ZXPSignCmd.exe>` signs it by hand), and compiles [`installer/FrameBlend.iss`](../installer/FrameBlend.iss). Output: `build/installer/FrameBlend-Setup-<version>.exe`, a per-machine installer (administrator rights) with an uninstaller under Settings › Apps. It installs:

- `FrameBlend.aex` and the four FFmpeg DLLs in `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\FrameBlend` (Premiere Pro, After Effects and Media Encoder 2020 and later load effects from there);
- the signed companion in `C:\Program Files (x86)\Common Files\Adobe\CEP\extensions\FrameBlend` (all users);
- `README.txt` and FFmpeg's LGPL licence in `C:\Program Files\Frame Blend`.

It refuses to run while Premiere Pro, After Effects or Media Encoder is open. The version is set in `installer/FrameBlend.iss` (`AppVersion`) and should match `PF_VERSION` in `plugins/adobe/frame_blend_ae.cpp` and the PiPL (`plugins/adobe/FrameBlendPiPL.r`).

## Building by hand

From the repository root in PowerShell:

```powershell
cargo build --release

$coreLib = Join-Path (Get-Location) 'target/release/frame_blend.lib'
cmake -S plugins/adobe -B build/adobe -DAE_SDK_PATH="$aeSdk" -DPR_SDK_PATH="$prSdk" -DFRAME_BLEND_CORE_LIB="$coreLib"
cmake --build build/adobe --config Release
```

Outputs: `build/adobe/Release/FrameBlend.aex` (Premiere Pro and After Effects) and the static core `target/release/frame_blend.lib` it links. `target/release/fb_render.exe` is an offline renderer using the same core (`fb_render --help`).

For development, `tools/install_plugins.ps1` (elevated) copies the build to the MediaCore folder, installs the companion for the current user only and creates `FrameBlend-log.txt` beside the plugin, which turns the debug log on. A release install writes no log: `plugins/common/fb_debug.cpp` only appends to a `FrameBlend-log.txt` that already exists beside the plugin or in `%TEMP%`.

The C and C++ runtime is linked statically (`.cargo/config.toml` sets `crt-static` for the core; the CMake files select `MultiThreaded`): hosts load their own `msvcp140.dll`, and Premiere Pro 2023 ships a 2015 version that crashes binaries built with current MSVC at their first `std::mutex` lock. With CUDA 12.0, `nvcc` predates MSVC 14.5; `plugins/common/gpu.cmake` passes `-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`, which is safe because the kernel uses no host standard library.

## Tests

- Core: `cargo test --release`.
- GPU parity (CUDA and OpenCL against the core's CPU path, 32- and 16-bit float, NV12 input) and the file reader:

  ```powershell
  cmake -S plugins/common/tests -B build/gpu_tests -DFRAME_BLEND_CORE_LIB="$coreLib"
  cmake --build build/gpu_tests --config Release
  build/gpu_tests/Release/gpu_parity.exe
  build/gpu_tests/Release/nv12_parity.exe
  build/gpu_tests/Release/media_reader_test.exe --pinned <numbered clips>
  build/gpu_tests/Release/media_reader_test.exe --pinned --vfr <variable frame rate recording>
  ```

  `tools/make_numbered_clips.py --output-dir <folder>` writes 240 and 600 fps test clips whose frames carry their own number (needs FFmpeg on the PATH).
