# AI FaceMat — Automated Face Matte Pipeline for Assimilate Scratch

An AI-powered pipeline that automatically generates per-feature face mattes (Skin, Lips, Eyes) from a selected shot in Assimilate Scratch, and loads them back as layers or versions for downstream compositing.

## Pipeline Overview

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Scratch    │───▶│  Render Frames   │───▶│  AI Analysis     │───▶│  Import Mattes   │
│  Shot Select │    │  (Snapshot API)  │    │  (GPU Pipeline)  │    │  (Layer/Version) │
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  1. SegFace Parsing  │  → Coarse semantic masks
                                    │  2. SAM2 Refinement  │  → Boundary-aware logits
                                    │  3. Video Tracking   │  → Temporal propagation
                                    └─────────────────────┘
```

## Quick Start

### Prerequisites
- **Windows** with NVIDIA GPU (CUDA)
- **Assimilate Scratch** running locally (`http://127.0.0.1:8080`)
- **Python 3.10+** with [uv](https://docs.astral.sh/uv/) package manager

### Launch via Batch File

```bat
run_AIFaceMat.bat -P1 1 -P2 y -P3 0
```

### Launch Directly

```bash
uv run AI_FaceMat.py -P1 1 -P2 y -P3 0
```

## Parameters

| Param | Values | Description |
|-------|--------|-------------|
| `-P1` | `0` = grayscale, `1` = color | **Output mode**: grayscale produces per-feature matte passes (Skin/Lips/Eyes as separate layers); color combines them into a single multicolor canvas (B=Skin, G=Eyes, R=Lips) |
| `-P2` | `y` / `n` | **Auto-clean**: removes rendered source cache after completion |
| `-P3` | `0` = layer, `1` = version | **Timeline destination**: layer adds matte overlays on the existing shot; version creates new shot entries |

### Common Recipes

```bat
:: Color multicolor mattes as layers
run_AIFaceMat.bat -P1 1 -P2 y -P3 0

:: Grayscale per-feature layers
run_AIFaceMat.bat -P1 0 -P2 y -P3 0

:: Color mattes as independent versions
run_AIFaceMat.bat -P1 1 -P2 n -P3 1
```

## Pipeline Stages

### Stage 1 — Metadata & Workspace
- Queries the active Scratch construct for the selected shot
- Reads the **timeline slot length** (not shot media length) for accurate trim range
- Resolves workspace paths from the project's cache/media configuration
- Calculates the trimmed timecode for matte conformance

### Stage 2 — Frame Rendering
- Renders the shot's trimmed frame range to disk via Scratch's Snapshot API (`POST /application/tools/image`)
- Uses the timeline slot length for frame count, respecting in/out handles
- Caches rendered frames — skips re-rendering if cached frames already exist

### Stage 3 — AI Analysis

#### 3a. SegFace Semantic Parsing
- Runs [SegFace](https://github.com/Kartik-3004/SegFace) (ConvNeXt backbone) on frame 0
- Produces a 19-class face segmentation map (CelebAMask-HQ classes)
- Extracts per-feature binary masks:
  - **Skin** → class index 2
  - **Lips** → class indices 12, 13 (lower lip, upper lip)
  - **Eyes** → class indices 8, 9 (left eye, right eye)

#### 3b. SAM2 Image Predictor Refinement
- Refines SegFace's coarse masks using [SAM 2.1](https://github.com/facebookresearch/sam2) image predictor with box prompts
- Each feature is refined independently with SAM2's boundary-aware decoder
- **Distance-field edge boosting** for thin features (Eyes, Lips):
  - Computes Euclidean distance transform from each pixel to the nearest background pixel
  - Applies exponential weight boost at mask edges (up to 2.5× for perimeter pixels)
  - Prevents thin features from being swallowed by neighboring classes during boundary conflicts

#### 3c. SAM2 Video Tracking
- Runs [SAM 2.1](https://github.com/facebookresearch/sam2) video predictor with refined masks as seed prompts
- `propagate_in_video` tracks all 3 features simultaneously through the shot
- Produces **binary masks** per frame
- Converts rendered PNGs to JPEGs (SAM2's frame loader requirement)
- Writes per-feature matte sequences to disk with diagnostic validation

#### 3d. Color Combine (color mode only)
- Reads per-feature alpha frames
- Composites into a multicolor canvas:
  - **Blue channel** = Skin
  - **Green channel** = Eyes
  - **Red channel** = Lips

### Stage 4 — Conformance
- Imports matte sequences back into Scratch
- **Layer mode** (`-P3 0`): Creates a matte layer on the existing shot with:
  - Correct slip offset for timeline alignment
  - Source timecode, fps, and reel_id conformed to the matte shot
- **Version mode** (`-P3 1`): Creates a new shot entry with source metadata
- Matte frame numbering aligns with the shot's trimmed range

### Stage 5 — Cleanup & Notifications
- Optionally purges rendered source cache (`-P2 y`)
- Appends a pipeline completion note to the shot's metadata

## Dependencies

| Package | Source | Purpose |
|---------|--------|---------|
| SegFace | [osmaras/SegFace](https://github.com/osmaras/SegFace) (fork) | Face semantic segmentation (ConvNeXt backbone) |
| SAM 2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Video object tracking + image predictor refinement |
| assimilate_client | [Assimilate-Inc/Assimilate-REST](https://github.com/Assimilate-Inc/Assimilate-REST) | Scratch REST API SDK |
| huggingface_hub | PyPI | Model checkpoint caching |

### Model Checkpoints

Checkpoints are auto-downloaded on first run and cached in `~/.cache/huggingface/hub/`:

| Model | HF Repo | File | ~Size |
|-------|---------|------|-------|
| SegFace | `kartiknarayan/SegFace` | `convnext_celeba_512/model_299.pt` | 350 MB |
| SAM 2.1 | `facebook/sam2.1-hiera-large` | `sam2.1_hiera_large.pt` | 856 MB |

## Workspace Structure

```
<project_cache>/Cache/AI_Pipeline_Workspace/<shot_uuid>/
├── source_sequence/          # Rendered source frames (PNG)
│   ├── 00005.png
│   ├── 00006.png
│   └── ...
├── masks/                    # SegFace + SAM2-refined masks
│   ├── skin.png
│   ├── lips.png
│   └── eyes.png
├── matte_skin/               # Per-feature alpha sequences
│   └── source_sequence/
│       └── pha/
│           ├── 00005.png
│           └── ...
├── matte_lips/
├── matte_eyes/
├── matte_multicolor/         # Combined color canvas (color mode)
└── sam2_frames/              # JPEG copies for SAM2 (SAM2 mode only)
```

## Assimilate Scratch API Endpoints Used

| Operation | Endpoint | Method |
|-----------|----------|--------|
| Get selected shot | `/constructs/current/sel_shots` | GET |
| Get slot properties | `/constructs/current/slots/{idx}` | GET |
| Get project paths | `/projects/current` | GET |
| Render frame snapshot | `/application/tools/image` | POST |
| Create shot | `/shot/new` | POST |
| Get/Set shot properties | `/shot/{uuid}` | GET / PUT |
| Create layer | `/shot/{uuid}/layers/new` | POST |
| Get layers | `/shot/{uuid}/layers` | GET |
| Set layer matte | `/shot/{uuid}/layers/{idx}/matte` | PUT |
| Add note | `/shot/{uuid}` (via PUT with notes array) | PUT |

## SegFace Class Mapping (CelebAMask-HQ)

| Index | Class | Used |
|-------|-------|------|
| 0 | Background | — |
| 1 | Neck | — |
| 2 | **Skin** | ✓ |
| 3 | Cloth | — |
| 4 | Left ear | — |
| 5 | Right ear | — |
| 6 | Left brow | — |
| 7 | Right brow | — |
| 8 | **Left eye** | ✓ |
| 9 | **Right eye** | ✓ |
| 10 | Nose | — |
| 11 | Mouth | — |
| 12 | **Lower lip** | ✓ |
| 13 | **Upper lip** | ✓ |
| 14 | Hair | — |
| 15 | Eyeglasses | — |
| 16 | Hat | — |
| 17 | Earring | — |
| 18 | Necklace | — |

## Notes

- The pipeline uses the **timeline slot length** (not the shot media length) for rendering, so trimmed shots are handled correctly.
- Matte timecode is calculated as `shot_frame_tc + in_point` to align with the trimmed timeline position.
- SAM2 produces binary masks — good for compositing workflows that need clean separations.
- The SAM2 image predictor refinement step runs on frame 0, improving boundary quality before temporal propagation.
- Distance-field edge boosting protects thin features (Eyes, Lips) from being swallowed by neighboring classes during boundary conflicts.
