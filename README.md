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

- **Windows 10/11** with NVIDIA GPU (CUDA 12.4+)
- **Assimilate Scratch** running locally (`http://127.0.0.1:8080`)

### Steps

1. **Clone the repository**

```bat
git clone https://github.com/osmaras/AI_FaceMat.git V:\PROGRAMING\Scratch-Scripts\AI_FaceMat
```

2. **Run the installer**

```powershell
powershell -File V:\PROGRAMING\Scratch-Scripts\AI_FaceMat\install.ps1
```

The installer will:
- Ask where to install (default: the cloned directory)
- Ask for a UV cache location (use a fast SSD like `G:\` if available)
- Check if [uv](https://docs.astral.sh/uv/) is installed — if not, install it automatically
- Patch `run_AIFaceMat.bat` with the correct paths and UV binary location
- Patch `Ai_Facemat.acc` with the correct batch file path for Scratch
- Generate an `uninstall.ps1` for clean removal

3. **Load the Custom Command in Scratch** (see [Configuration](#configuration) below)

4. **First run** — uv will automatically install Python 3.10+, PyTorch with CUDA, and all AI model dependencies

> On first run, model checkpoints (~1.2 GB total) are downloaded from HuggingFace and cached in `~/.cache/huggingface/hub/`. Subsequent runs use the cache.

### Uninstall

```powershell
powershell -File <install_dir>\uninstall.ps1
```

The uninstaller asks before removing the install directory and UV cache (models + environments). HuggingFace checkpoint cache is preserved separately.

## Configuration

### Load the Custom Command in Scratch

The pipeline runs as a **Custom Command** inside Scratch. Custom Commands extend the application with user-defined scripts and actions — they appear as buttons in the construct or player menus, and all user input values are passed to the script through command-line parameters. Commands can be saved and loaded as standalone `.acc` files.

1. Open **Scratch** and load your project
2. Go to **System Settings → Custom Commands → Import**
3. Browse to the `Ai_Facemat.acc` file in your install directory
4. The **Ai_FaceMat** button will appear in the player right-click menu (only when a shot is selected)

> If you ran `install.ps1`, the `.acc` file is already patched with the correct paths. If you moved the installation, re-run the installer or manually edit the `<cmdline>` path in `Ai_Facemat.acc`.

### Input Form

The custom command presents three inputs when triggered. All values are passed to the script through command-line parameters:

| Input | Type | Options | Script Parameter |
|-------|------|---------|-----------------|
| **Processing Mode** | Dropdown | `grayscale` / `color` | `-P1` (0 / 1) |
| **Clean up cache** | Yes/No | Checkbox | `-P2` (y / n) |
| **Timeline destination** | Dropdown | `Add layer` / `Add version` | `-P3` (0 / 1) |
| **SegFace Model** | Dropdown | `ConvNeXt CelebA` / `SwinB CelebA` / `SwinV2B CelebA` / `SwinB LaPa` | `-P4` (0 / 1 / 2 / 3) |

## Usage

### Basic Workflow

1. **Select a shot** in the Scratch timeline (the button only appears when a shot is selected)
2. **Right-click** the player → select **Ai_FaceMat**
3. **Choose options** in the input form:
   - **Processing Mode**: `color` for a single multicolor matte, `grayscale` for individual per-feature mattes
   - **Clean up cache**: `yes` to remove rendered frames after completion
   - **Timeline destination**: `Add layer` to overlay on the existing shot, `Add version` to create a new shot entry
   - **SegFace Model**: `ConvNeXt` for highest accuracy, `SwinV2B` for best F1 score, `SwinB` for speed, `SwinB LaPa` for alternative dataset quality
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

## Project Structure

| File | Description |
|------|-------------|
| `AI_FaceMat.py` | Main pipeline script (SegFace + SAM2 + Scratch integration) |
| `scratch_api.py` | Reusable Scratch REST API wrapper — importable from other scripts |
| `run_AIFaceMat.bat` | Batch launcher (configured by installer) |
| `Ai_Facemat.acc` | Scratch Custom Command definition (load in Scratch via System Settings → Custom Commands → Import) |
| `install.ps1` | Interactive installer (uv detection, path config, .acc/.bat patching) |

### Reusing the Scratch API

`scratch_api.py` can be imported independently in any Python script that needs to interact with Scratch:

```python
from scratch_api import ScratchAPI

scratch = ScratchAPI()
selection, shot_data = scratch.get_selected_shot()
print(f"Shot: {shot_data.name}, Length: {shot_data.length}")
```

## Dependencies

| Package | Source | Purpose |
|---------|--------|---------|
| SegFace | [osmaras/SegFace](https://github.com/osmaras/SegFace) | Face semantic segmentation (ConvNeXt backbone) |
| SAM 2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Video object tracking + image predictor refinement |
| assimilate_client | [Assimilate-Inc/Assimilate-REST](https://github.com/Assimilate-Inc/Assimilate-REST) | Scratch REST API SDK |
| huggingface_hub | PyPI | Model checkpoint caching |

### Model Checkpoints

Auto-downloaded on first run, cached in `~/.cache/huggingface/hub/`:

| `-P4` | Model | Backbone | Dataset | HF File | Size |
|-------|-------|----------|---------|---------|------|
| 0 | ConvNeXt CelebA | `convnext_base` | CelebAMask-HQ | `convnext_celeba_512/model_299.pt` | ~350 MB |
| 1 | SwinB CelebA | `swin_base` | CelebAMask-HQ | `swinb_celeba_512/model_299.pt` | ~350 MB |
| 2 | SwinV2B CelebA | `swinv2_base` | CelebAMask-HQ | `swinv2b_celeba_512/model_299.pt` | ~350 MB |
| 3 | SwinB LaPa | `swin_base` | LaPa | `swinb_lapa_512/model_299.pt` | ~350 MB |
| — | SAM 2.1 | — | — | `sam2.1_hiera_large.pt` | ~856 MB |

All SegFace checkpoints are from `kartiknarayan/SegFace`. SAM 2.1 is from `facebook/sam2.1-hiera-large`.

**Model selection guide:**
- **ConvNeXt CelebA** (`-P4 0`): Highest accuracy across all classes, especially long-tail features (earrings, eyeglasses, necklaces). Slower inference.
- **SwinB CelebA** (`-P4 1`): Faster inference with Swin Transformer backbone. Slightly lower accuracy on rare classes.
- **SwinV2B CelebA** (`-P4 2`): SwinV2 backbone — best overall F1 score on CelebAMask-HQ (88.73). Good balance of speed and accuracy.
- **SwinB LaPa** (`-P4 3`): Trained on the LaPa dataset which has cleaner face boundaries and more consistent annotations. Uses different class indices (11 classes vs 19). Good alternative if CelebA results are unsatisfactory.

## SegFace Class Mapping

The pipeline maps SegFace's semantic output to three features (Skin, Lips, Eyes). The class indices differ between datasets:

### CelebAMask-HQ (19 classes) — used by ConvNeXt, SwinB, SwinV2B

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

### LaPa (11 classes) — used by SwinB LaPa

| Index | Class | Used | | Index | Class | Used |
|-------|-------|------|-|-------|-------|------|
| 0 | Background | — | | 6 | Nose | — |
| 1 | **Face** | ✓ (Skin) | | 7 | **Upper lip** | ✓ |
| 2 | Left brow | — | | 8 | Inner mouth | — |
| 3 | Right brow | — | | 9 | **Lower lip** | ✓ |
| 4 | **Left eye** | ✓ | | 10 | Hair | — |
| 5 | **Right eye** | ✓ | | | | |

### Adding Custom Feature Groups

To extract additional face features (e.g., Hair, Nose, Eyeglasses), edit the `feature_masks` dictionary in `AI_FaceMat.py` (Stage 3 section):

```python
feature_masks = {
    "Skin": (parsed_semantic_map == 2).astype(np.uint8) * 255,
    "Lips": np.isin(parsed_semantic_map, [12, 13]).astype(np.uint8) * 255,
    "Eyes": np.isin(parsed_semantic_map, [8, 9]).astype(np.uint8) * 255,
    # Add more features here:
    "Hair": (parsed_semantic_map == 14).astype(np.uint8) * 255,
    "Nose": (parsed_semantic_map == 10).astype(np.uint8) * 255,
    "Eyeglasses": (parsed_semantic_map == 15).astype(np.uint8) * 255,
}
```

Then add the new feature names to the `FEATURES` list:

```python
FEATURES = ["Skin", "Lips", "Eyes", "Hair", "Nose", "Eyeglasses"]
```

For multi-class features that span multiple indices (like Lips = 12 + 13), use `np.isin()`. For single-class features, use `==`.

The SAM2 refinement and video tracking will automatically pick up the new features — no other code changes needed.
