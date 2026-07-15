# AI FaceMat — Automated Face Matte Pipeline for Assimilate Scratch

An AI-powered pipeline that automatically generates per-feature face mattes (Skin, Lips, Eyes) from a selected shot in Assimilate Scratch, and loads them back as layers or versions for downstream compositing.

```
┌──────────────┐      ┌──────────────────┐     ┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│   Scratch    │───▶ │  Render Frames    │───▶│  SegFace Parse  │───▶│  SAM2 Refine +    │───▶│ Import Mattes      │
│  Shot Select │      │  (API)           │     │  (GPU, frame 0) │     │  Video Tracking   │     │  (Layer/Version)  │
└──────────────┘      └──────────────────┘     └─────────────────┘     └───────────────────┘     └───────────────────┘
```

## Installation

### Prerequisites

- **Windows 10/11** with NVIDIA GPU
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
- **Auto-detect your GPU** via `nvidia-smi` and select the correct CUDA/PyTorch build from `torch-versions.json`
- Patch `run_AIFaceMat.bat` with the correct paths, UV binary, and PyTorch index
- Patch `AI_FaceMat.py` with the detected torch/torchvision versions
- Patch `Ai_Facemat.acc` with the correct batch file path for Scratch

**GPU auto-detection:**

| Compute Capability | GPU Family | CUDA Build | PyTorch |
|---|---|---|---|
| sm < 10.0 | Ampere, Ada Lovelace, Hopper | cu124 | torch 2.5.1 |
| sm ≥ 10.0 | Blackwell (B100, B200, RTX 50xx) | cu128 | torch 2.7.0 |
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

The custom command presents four inputs when triggered. All values are passed to the script through command-line parameters:

| Input | Type | Options | Script Parameter |
|-------|------|---------|-----------------|
| **Processing Mode** | Dropdown | `grayscale` / `color` | `-P1` (0 / 1) |
| **Clean up cache** | Yes/No | Checkbox | `-P2` (y / n) |
| **Timeline destination** | Dropdown | `Add layer` / `Add version` | `-P3` (0 / 1) |
| **SegFace Model** | Dropdown | `ConvNeXt CelebA` / `SwinB CelebA` / `SwinV2B CelebA` / `SwinB LaPa` | `-P4` (0 / 1 / 2 / 3) |

### Scratch REST API Connection

The pipeline communicates with Scratch via its REST API. The connection settings are passed as environment variables in `run_AIFaceMat.bat`:

```bat
set "SCRATCH_PORT=8080"
set "SCRATCH_API_KEY="
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRATCH_PORT` | `8080` | REST API port configured in Scratch |
| `SCRATCH_API_KEY` | *(empty)* | Optional access key for authenticated Scratch instances |

> **If you change the REST API port in Scratch** (System Settings → REST API → Port), update `SCRATCH_PORT` in `run_AIFaceMat.bat` to match. If the port doesn't match, the pipeline will fail to connect.

The installer prompts for these values during setup. To change them later, edit `run_AIFaceMat.bat` directly or re-run `install.ps1`.

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

Each stage is a class in `AI_FaceMat.py`, orchestrated by the `Pipeline` class. Shared state flows through a `PipelineContext` dataclass.

### Stage 1 — Metadata & Workspace (`MetadataStage`)
- Queries the active Scratch construct for the selected shot
- Reads the **timeline slot length** (not shot media length) for accurate trim range
- Resolves workspace paths from the project's cache/media configuration
- Creates a `FrameMapper` for frame numbering alignment

### Stage 2 — Frame Rendering (`RenderStage`)
- Renders the shot's trimmed frame range to disk via Scratch's render shot API
- Uses the timeline slot length for frame count, respecting in/out handles
- Caches rendered frames — skips re-rendering if cached frames already exist

### Stage 3 — SegFace Semantic Parsing (`SegFaceStage`)
- Runs [SegFace](https://github.com/Kartik-3004/SegFace) on frame 0
- Produces a per-dataset face segmentation map (19 classes for CelebAMask-HQ, 11 for LaPa)
- Extracts per-feature binary masks (Skin, Lips, Eyes) via `FEATURE_CLASS_MAP`

### Stage 4 — SAM2 Refinement & Tracking (`SAM2Stage`)
- **Image predictor**: refines SegFace's coarse masks using [SAM 2.1](https://github.com/facebookresearch/sam2) with box prompts
- **Distance-field edge boosting** for thin features (Eyes, Lips): prevents them from being swallowed by neighboring classes during boundary conflicts
- **Video tracking**: propagates refined masks through the entire shot via `propagate_in_video`
- **Color combine** (color mode only): composites per-feature alphas into a multicolor canvas

### Stage 5 — Conform (`ConformStage`)
- Imports matte sequences back into Scratch as layers or versions
- Injects source metadata (timecode, fps, reel_id) onto matte shots
- Uses `FrameMapper` to align matte frame numbering with the shot's trimmed range

### Stage 6 — Cleanup & Notifications (`CleanupStage`)
- Optionally purges rendered source cache
- Appends a pipeline completion note to the shot's metadata

## Project Structure

| File | Description |
|------|-------------|
| `AI_FaceMat.py` | Main pipeline: 6 stage classes, `Pipeline` orchestrator, `FrameMapper`, `PipelineContext`, SegFace + SAM2 + Scratch integration |
| `scratch_api.py` | Reusable Scratch REST API wrapper — importable from other scripts |
| `torch-versions.json` | Single source of truth for PyTorch/CUDA version mappings (read by `install.ps1`) |
| `run_AIFaceMat.bat` | Batch launcher (configured by installer) |
| `Ai_Facemat.acc` | Scratch Custom Command definition (load in Scratch via System Settings → Custom Commands → Import) |
| `install.ps1` | Interactive installer (uv detection, GPU detection, path config, .acc/.bat/.py patching) |

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

PyTorch and torchvision versions are managed via `torch-versions.json` (single source of truth). The installer reads this file and patches `AI_FaceMat.py` accordingly. To update PyTorch, edit the JSON file only.

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

To extract additional face features (e.g., Hair, Nose, Eyeglasses):

1. Add the feature to `FEATURE_CLASS_MAP` in `AI_FaceMat.py` for each dataset you use:

```python
FEATURE_CLASS_MAP = {
    "celeba": {
        "Skin": lambda m: (m == 2).astype(np.uint8) * 255,
        "Lips": lambda m: np.isin(m, [12, 13]).astype(np.uint8) * 255,
        "Eyes": lambda m: np.isin(m, [8, 9]).astype(np.uint8) * 255,
        # Add more features here:
        "Hair": lambda m: (m == 14).astype(np.uint8) * 255,
        "Nose": lambda m: (m == 10).astype(np.uint8) * 255,
        "Eyeglasses": lambda m: (m == 15).astype(np.uint8) * 255,
    },
    "lapa": { ... },
}
```

2. Add the feature name to the `FEATURES` list:

```python
FEATURES = ["Skin", "Lips", "Eyes", "Hair", "Nose", "Eyeglasses"]
```

3. Add a SAM2 `obj_id` mapping in `SAM2Stage.run()` (must be unique integers):

```python
sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=4, mask=ctx.feature_masks["Hair"])
```

And update `obj_id_to_feature`:

```python
obj_id_to_feature = {1: "Lips", 2: "Skin", 3: "Eyes", 4: "Hair"}
```

For multi-class features that span multiple indices (like Lips = 12 + 13), use `np.isin()`. For single-class features, use `==`.
