# AI FaceMat — Automated Face Matte Pipeline for Assimilate Scratch

An AI-powered pipeline that automatically generates per-feature face mattes (Skin, Lips, Eyes) from a selected shot in Assimilate Scratch, and loads them back as layers or versions for downstream compositing.

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

## Installation

### Prerequisites

- **Windows** with NVIDIA GPU (CUDA 12.4+)
- **Assimilate Scratch** running locally (`http://127.0.0.1:8080`)
- [uv](https://docs.astral.sh/uv/) package manager (handles Python and all dependencies automatically)

### Steps

1. **Clone the repository**

```bat
git clone https://github.com/osmaras/AI_FaceMat.git V:\PROGRAMING\Scratch-Scripts\AI_FaceMat
```

2. **Verify uv is installed**

```bat
uv --version
```

If not installed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **First run** — uv will automatically install Python 3.10+, PyTorch with CUDA, and all AI model dependencies:

```bat
run_AIFaceMat.bat -P1 1 -P2 y -P3 0
```

> On first run, model checkpoints (~1.2 GB total) are downloaded from HuggingFace and cached in `~/.cache/huggingface/hub/`. Subsequent runs use the cache.

## Configuration

### Load the Custom Command in Scratch

The pipeline runs as a **Custom Command** inside Scratch. Custom Commands extend the application with user-defined scripts and actions — they appear as buttons in the construct or player menus, and all user input values are passed to the script through command-line parameters. Commands can be saved and loaded as standalone `.acc` files.

1. Open **Scratch** and load your project
2. Go to **Menu → Custom Commands → Load**
3. Browse to `V:\PROGRAMING\Scratch-Scripts\AI_FaceMat\Ai_Facemat.acc`
4. The **Ai_FaceMat** button will appear in the player right-click menu (only when a shot is selected)

### Customize the Command Path

If you installed the repository to a different location, edit the `<cmdline>` path in `Ai_Facemat.acc`:

```xml
<cmdline>V:\YOUR\PATH\TO\AI_FaceMat\run_AIFaceMat.bat</cmdline>
```

### Input Form

The custom command presents three inputs when triggered. All values are passed to the script through command-line parameters:

| Input | Type | Options | Script Parameter |
|-------|------|---------|-----------------|
| **Processing Mode** | Dropdown | `grayscale` / `color` | `-P1` (0 / 1) |
| **Clean up cache** | Yes/No | Checkbox | `-P2` (y / n) |
| **Timeline destination** | Dropdown | `Add layer` / `Add version` | `-P3` (0 / 1) |

## Usage

### Basic Workflow

1. **Select a shot** in the Scratch timeline (the button only appears when a shot is selected)
2. **Right-click** the player → select **Ai_FaceMat**
3. **Choose options** in the input form:
   - **Processing Mode**: `color` for a single multicolor matte, `grayscale` for individual per-feature mattes
   - **Clean up cache**: `yes` to remove rendered frames after completion
   - **Timeline destination**: `Add layer` to overlay on the existing shot, `Add version` to create a new shot entry
4. **Click OK** — the pipeline renders frames, runs AI analysis, and imports mattes back into Scratch

### Output Modes

| Mode | Description | Scratch Result |
|------|-------------|----------------|
| **Color** | Single multicolor canvas (B=Skin, G=Eyes, R=Lips) | One matte layer/shot |
| **Grayscale** | Individual binary masks per feature | Three matte layers/shots (Skin, Lips, Eyes) |

### Timeline Destinations

| Destination | Description |
|-------------|-------------|
| **Add layer** | Creates a matte layer on the existing shot with correct slip offset for timeline alignment. Source timecode, fps, and reel_id are conformed to the matte. |
| **Add version** | Creates a new shot entry in the Scratch library with source metadata (timecode, fps, reel_id, clip name). |

### Cache Behavior

- Rendered source frames are cached in the project's cache directory
- Subsequent runs on the same shot skip the render pass if cached frames exist
- Enable **Clean up cache** to reclaim disk space after completion

## Pipeline Stages

### Stage 1 — Metadata & Workspace
- Queries the active Scratch construct for the selected shot
- Reads the **timeline slot length** (not shot media length) for accurate trim range
- Resolves workspace paths from the project's cache/media configuration
- Calculates the trimmed timecode for matte conformance

### Stage 2 — Frame Rendering
- Renders the shot's trimmed frame range to disk via Scratch's Snapshot API
- Uses the timeline slot length for frame count, respecting in/out handles
- Caches rendered frames — skips re-rendering if cached frames already exist

### Stage 3 — AI Analysis

#### SegFace Semantic Parsing
- Runs [SegFace](https://github.com/Kartik-3004/SegFace) (ConvNeXt backbone) on frame 0
- Produces a 19-class face segmentation map (CelebAMask-HQ)
- Extracts per-feature binary masks (Skin, Lips, Eyes)

#### SAM2 Image Predictor Refinement
- Refines SegFace's coarse masks using [SAM 2.1](https://github.com/facebookresearch/sam2) with box prompts
- **Distance-field edge boosting** for thin features (Eyes, Lips): prevents them from being swallowed by neighboring classes during boundary conflicts

#### SAM2 Video Tracking
- Runs SAM 2.1 video predictor with refined masks as seed prompts
- `propagate_in_video` tracks all 3 features simultaneously through the shot
- Produces **binary masks** per frame

#### Color Combine (color mode only)
- Composites per-feature alphas into a multicolor canvas

### Stage 4 — Conformance
- Imports matte sequences back into Scratch as layers or versions
- Conforms source metadata (timecode, fps, reel_id) onto matte shots
- Aligns matte frame numbering with the shot's trimmed range

### Stage 5 — Cleanup & Notifications
- Optionally purges rendered source cache
- Appends a pipeline completion note to the shot's metadata

## Dependencies

| Package | Source | Purpose |
|---------|--------|---------|
| SegFace | [osmaras/SegFace](https://github.com/osmaras/SegFace) | Face semantic segmentation (ConvNeXt backbone) |
| SAM 2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Video object tracking + image predictor refinement |
| assimilate_client | [Assimilate-Inc/Assimilate-REST](https://github.com/Assimilate-Inc/Assimilate-REST) | Scratch REST API SDK |
| huggingface_hub | PyPI | Model checkpoint caching |

### Model Checkpoints

Auto-downloaded on first run, cached in `~/.cache/huggingface/hub/`:

| Model | HF Repo | File | Size |
|-------|---------|------|------|
| SegFace | `kartiknarayan/SegFace` | `convnext_celeba_512/model_299.pt` | ~350 MB |
| SAM 2.1 | `facebook/sam2.1-hiera-large` | `sam2.1_hiera_large.pt` | ~856 MB |

## SegFace Class Mapping (CelebAMask-HQ)

| Index | Class | Used | | Index | Class | Used |
|-------|-------|------|-|-------|-------|------|
| 0 | Background | — | | 10 | Nose | — |
| 1 | Neck | — | | 11 | Mouth | — |
| 2 | **Skin** | ✓ | | 12 | **Lower lip** | ✓ |
| 3 | Cloth | — | | 13 | **Upper lip** | ✓ |
| 4 | Left ear | — | | 14 | Hair | — |
| 5 | Right ear | — | | 15 | Eyeglasses | — |
| 6 | Left brow | — | | 16 | Hat | — |
| 7 | Right brow | — | | 17 | Earring | — |
| 8 | **Left eye** | ✓ | | 18 | Necklace | — |
| 9 | **Right eye** | ✓ | | | | |
