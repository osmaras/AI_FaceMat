#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python>=4.8.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "torch==2.7.0+cu128",
#     "torchvision==0.22.0+cu128",
#     "sam-2 @ git+https://github.com/facebookresearch/sam2.git",
#     "segface @ git+https://github.com/osmaras/SegFace.git",
#     "assimilate_client @ git+https://github.com/Assimilate-Inc/Assimilate-REST.git",
#     "huggingface_hub>=0.20.0",
#   
# ]
# ///

import os
import sys
import time
import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

# --- MODEL CHECKPOINT RESOLUTION ---
# Uses huggingface_hub which checks ~/.cache/huggingface/hub first (instant
# if already downloaded), then downloads only on cache miss.
CHECKPOINTS = {
    "segface_convnext": {
        "repo_id": "kartiknarayan/SegFace",
        "filename": "convnext_celeba_512/model_299.pt",
        "backbone": "segface_celeb",
        "model": "convnext_base",
        "dataset": "celeba",
    },
    "segface_swinb": {
        "repo_id": "kartiknarayan/SegFace",
        "filename": "swinb_celeba_512/model_299.pt",
        "backbone": "segface_celeb",
        "model": "swin_base",
        "dataset": "celeba",
    },
    "segface_swinv2b": {
        "repo_id": "kartiknarayan/SegFace",
        "filename": "swinv2b_celeba_512/model_299.pt",
        "backbone": "segface_celeb",
        "model": "swinv2_base",
        "dataset": "celeba",
    },
    "segface_swinb_lapa": {
        "repo_id": "kartiknarayan/SegFace",
        "filename": "swinb_lapa_512/model_299.pt",
        "backbone": "segface_lapa",
        "model": "swin_base",
        "dataset": "lapa",
    },
    "sam2": {
        "repo_id": "facebook/sam2.1-hiera-large",
        "filename": "sam2.1_hiera_large.pt",
    },
}

# Per-dataset class index mappings for feature extraction
# CelebAMask-HQ (19 classes): background, neck, skin, cloth, l_ear, r_ear,
#   l_brow, r_brow, l_eye, r_eye, nose, mouth, l_lip, u_lip, hair, ...
# LaPa (11 classes): background, face, lb, rb, le, re, nose, ul, im, ll, hair
FEATURE_CLASS_MAP = {
    "celeba": {
        "Skin": lambda m: (m == 2).astype(np.uint8) * 255,
        "Lips": lambda m: np.isin(m, [12, 13]).astype(np.uint8) * 255,
        "Eyes": lambda m: np.isin(m, [8, 9]).astype(np.uint8) * 255,
    },
    "lapa": {
        "Skin": lambda m: (m == 1).astype(np.uint8) * 255,   # face region
        "Lips": lambda m: np.isin(m, [7, 9]).astype(np.uint8) * 255,  # upper lip + lower lip
        "Eyes": lambda m: np.isin(m, [4, 5]).astype(np.uint8) * 255,  # left eye + right eye
    },
}

FEATURES = ["Skin", "Lips", "Eyes"]


def ensure_checkpoint(name):
    """
    Resolve a model checkpoint path via the HuggingFace cache.
    Returns instantly if the model is already cached locally;
    downloads on first run only.
    """
    spec = CHECKPOINTS[name]
    print(f"  🔍 Resolving {name} checkpoint from HF cache...")
    path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"])
    print(f"  ✓ {name}: {path}")
    return path

# --- MODEL ENTRYPOINTS ---
from segface.inference import SegFaceParser
# SAM2 is imported lazily (only when needed) to avoid loading it unnecessarily


class FrameMapper:
    """Owns frame_tc, in_point, and sequence↔timecode conversions.

    Strict: raises if frame_tc or handles are missing, so callers never
    silently get wrong alignment.
    """

    def __init__(self, shot_data, selection=None):
        frame_tc = shot_data.frame_tc
        if frame_tc is None:
            raise ValueError("shot_data.frame_tc is None — cannot determine frame timecode")
        self._frame_tc = int(frame_tc)

        handles = getattr(shot_data, "handles", None)
        if handles and handles.frame_in is not None:
            self._in_point = int(handles.frame_in)
        else:
            self._in_point = self._frame_tc

    def to_sequence(self, idx):
        """Map a zero-based propagation index to the actual frame number."""
        return self._in_point + idx

    def to_timecode(self):
        """Return the matte start timecode (shot TC + in_point offset)."""
        return self._frame_tc + self._in_point

    @property
    def slip_offset(self):
        """Slip value for matte layer alignment."""
        return self._in_point


def refine_masks_sam2_image(image_bgr, feature_masks, sam2_ckpt, device="cuda"):
    """
    Refine SegFace coarse masks using SAM2 image predictor with box prompts.
    Each feature mask is refined independently via SAM2's boundary-aware logits.
    Thin features (Eyes, Lips) get distance-field edge boosting to prevent
    them from being swallowed by neighboring classes.

    Runs only on the keyframe (frame 0) — the tracker handles the rest.
    """
    from sam2.build_sam import build_sam2 as _build_sam2_img
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from torchvision.ops import masks_to_boxes
    from scipy.ndimage import distance_transform_edt

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_bgr.shape[:2]

    print("  🔧 SAM2 image predictor: refining mask boundaries...")
    predictor = SAM2ImagePredictor(
        _build_sam2_img("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_ckpt, device=device)
    )
    predictor.set_image(image_rgb)

    # Thin features get distance-field edge protection
    protected_features = {"Eyes", "Lips"}

    refined = {}
    for name, mask in feature_masks.items():
        binary = (mask > 127).astype(np.uint8)
        if not np.any(binary):
            print(f"    {name}: empty mask, skipping refinement")
            refined[name] = mask
            continue

        # Box prompt from SegFace mask
        mask_tensor = torch.from_numpy(binary > 0).unsqueeze(0)
        bbox = masks_to_boxes(mask_tensor).numpy()

        # SAM2 image prediction with box prompt → logits
        _, _, low_res_logits = predictor.predict(
            box=bbox, multimask_output=False, return_logits=True
        )
        high_res_logits = F.interpolate(
            torch.from_numpy(low_res_logits).unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze()

        # Distance-field edge boost for thin features
        if name in protected_features:
            dist_field = distance_transform_edt(binary)
            max_dist = dist_field.max()
            if max_dist > 0:
                norm_dist = dist_field / max_dist
                edge_boost = 1.50 * np.exp(-4.0 * norm_dist)
                weight = torch.from_numpy(1.0 + edge_boost).float()
                high_res_logits = high_res_logits * weight

        # Threshold refined logits
        refined_mask = (high_res_logits > 0.0).cpu().numpy().astype(np.uint8) * 255
        orig_nz = np.count_nonzero(binary)
        ref_nz = np.count_nonzero(refined_mask)
        print(f"    {name}: {orig_nz} → {ref_nz} nonzero pixels")
        refined[name] = refined_mask

    # Clean up
    del predictor
    torch.cuda.empty_cache()
    return refined

# --- ASSIMILATE SDK ENTRYPOINTS ---
from scratch_api import ScratchAPI


# ==============================================================================
# PIPELINE CONTEXT — shared state passed between stages
# ==============================================================================
@dataclass
class PipelineContext:
    """Holds all shared state for the pipeline stages."""
    args: argparse.Namespace
    scratch: ScratchAPI
    # Shot metadata (set by MetadataStage)
    shot_uuid: str = ""
    shot_name: str = ""
    frame_mapper: Optional[FrameMapper] = None
    shot_length: int = 1
    shot_data: object = None
    source_tc: str = "00:00:00:00"
    source_reel: str = "A001C001"
    # Workspace paths (set by MetadataStage)
    pipeline_workspace: str = ""
    render_in: str = ""
    matte_dirs: dict = field(default_factory=dict)
    # Render output (set by RenderStage)
    frame_list: list = field(default_factory=list)
    # SegFace output (set by SegFaceStage)
    h: int = 0
    w: int = 0
    feature_masks: dict = field(default_factory=dict)
    mask_dir: str = ""
    # SAM2 output (set by SAM2Stage)
    alpha_dirs: dict = field(default_factory=dict)


# ==============================================================================
# PIPELINE STAGES
# ==============================================================================

class MetadataStage:
    """Stage 1: Fetch shot metadata and set up workspace directories."""

    def run(self, ctx: PipelineContext) -> PipelineContext:
        selection, shot_data = ctx.scratch.get_selected_shot()
        if not selection or not shot_data:
            print("❌ No shot selected in the current construct. Aborting.")
            sys.exit(1)

        ctx.shot_data = shot_data
        ctx.shot_uuid = str(selection.uuid)
        ctx.shot_name = shot_data.name or "VFX_Shot"
        ctx.frame_mapper = FrameMapper(shot_data)

        slot_idx = int(getattr(selection, "slot_idx", 0) or 0)
        slot_data = ctx.scratch.get_slot(slot_idx)
        slot_length = int(getattr(slot_data, "length", 0) or 0)
        ctx.shot_length = slot_length if slot_length > 0 else (shot_data.length or 1)
        print(f"  ↳ Slot {slot_idx} length: {slot_length} | Shot media length: {shot_data.length}")

        ctx.source_tc = str(shot_data.timecode) if shot_data.timecode else "00:00:00:00"
        ctx.source_reel = shot_data.reel_id or "A001C001"

        proj_paths = ctx.scratch.get_project_paths()
        if proj_paths and proj_paths.cache_path:
            workspace_base = proj_paths.cache_path
        elif proj_paths and proj_paths.media_path:
            workspace_base = proj_paths.media_path
        elif shot_data.file:
            workspace_base = os.path.dirname(shot_data.file)
        else:
            workspace_base = os.getcwd()

        ctx.pipeline_workspace = os.path.join(workspace_base, "Cache", "AI_Pipeline_Workspace", ctx.shot_uuid)
        ctx.render_in = os.path.join(ctx.pipeline_workspace, "source_sequence")
        os.makedirs(ctx.render_in, exist_ok=True)

        if ctx.args.mode == "grayscale":
            for feature in FEATURES:
                ctx.matte_dirs[feature] = os.path.join(ctx.pipeline_workspace, f"matte_{feature.lower()}")
                os.makedirs(ctx.matte_dirs[feature], exist_ok=True)
        else:
            ctx.matte_dirs["Combined"] = os.path.join(ctx.pipeline_workspace, "matte_multicolor")
            os.makedirs(ctx.matte_dirs["Combined"], exist_ok=True)

        print(f"🎬 Active Shot: '{ctx.shot_name}' [UUID: {ctx.shot_uuid}]")
        print(f"📂 Workspace: {ctx.pipeline_workspace}")
        print(f"🎞️ Frame Range: {ctx.frame_mapper.slip_offset} + {ctx.shot_length} frames")
        return ctx


class RenderStage:
    """Stage 2: Render frames to disk via Scratch render shot."""

    def run(self, ctx: PipelineContext) -> PipelineContext:
        existing_frames = [f for f in os.listdir(ctx.render_in) if f.endswith('.jpg')]
        if len(existing_frames) >= ctx.shot_length:
            print(f"🎞️ Found {len(existing_frames)} cached frames, skipping render pass.")
        else:
            print(f"🎞️ Rendering {ctx.shot_length} frames via render shot...")
            render_node = ctx.scratch.create_render_shot(
                name=f"AI_Pipeline_{ctx.shot_name}",
                filespec="#frame[6].#ext",
                outputpath=ctx.render_in,
                input_shot_uuid=ctx.shot_uuid,
                file_format="jpg",
            )
            render_uuid = str(render_node.uuid)
            try:
                queue_item = ctx.scratch.start_render(render_uuid, delete_existing_media=ctx.args.auto_clean)
                print("  ↳ Rendering", end="", flush=True)
                while queue_item.status in ("waiting", "processing"):
                    time.sleep(1)
                    print(".", end="", flush=True)
                    queue_item = ctx.scratch.poll_render(render_uuid)
                print()
                if queue_item.status != "finished":
                    err_detail = getattr(queue_item, "error", None) or getattr(queue_item, "message", None)
                    print(f"❌ Render failed with status: {queue_item.status}"
                          + (f" ({err_detail})" if err_detail else ""))
                    sys.exit(1)
            finally:
                ctx.scratch.delete_render_shot(render_uuid)

        ctx.frame_list = sorted([f for f in os.listdir(ctx.render_in) if f.endswith('.jpg')])
        if not ctx.frame_list:
            print("❌ Error: Render cache folder is empty. Cannot continue.")
            sys.exit(1)
        return ctx


class SegFaceStage:
    """Stage 3: SegFace semantic parsing on keyframe (frame 0)."""

    def run(self, ctx: PipelineContext) -> PipelineContext:
        print("🧠 Initializing deep learning models on GPU...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        segface_spec = CHECKPOINTS[ctx.args.segface_model]
        segface_ckpt = ensure_checkpoint(ctx.args.segface_model)
        print(f"  ↳ Using SegFace model: {ctx.args.segface_model} ({segface_spec['model']})")
        face_parser = SegFaceParser(
            checkpoint=segface_ckpt,
            device=device,
            backbone=segface_spec["backbone"],
            model_name=segface_spec["model"],
        )

        sample_img = cv2.imread(os.path.join(ctx.render_in, ctx.frame_list[0]))
        ctx.h, ctx.w, _ = sample_img.shape

        print("🎭 Running SegFace semantic parsing on frame 0...")
        parsed_semantic_map = face_parser.parse_image(sample_img)

        dataset = segface_spec.get("dataset", "celeba")
        class_map = FEATURE_CLASS_MAP[dataset]
        print(f"  ↳ Dataset: {dataset} ({len(class_map)} features)")

        ctx.mask_dir = os.path.join(ctx.pipeline_workspace, "masks")
        os.makedirs(ctx.mask_dir, exist_ok=True)
        ctx.feature_masks = {name: fn(parsed_semantic_map) for name, fn in class_map.items()}
        for name, mask_arr in ctx.feature_masks.items():
            cv2.imwrite(os.path.join(ctx.mask_dir, f"{name.lower()}.png"), mask_arr)
            print(f"    {name}: {np.count_nonzero(mask_arr)} nonzero / {mask_arr.size} total pixels")

        return ctx


class SAM2Stage:
    """Stage 4: SAM2 image refinement + video tracking."""

    def run(self, ctx: PipelineContext) -> PipelineContext:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_ckpt = ensure_checkpoint("sam2")

        sample_img = cv2.imread(os.path.join(ctx.render_in, ctx.frame_list[0]))
        ctx.feature_masks = refine_masks_sam2_image(sample_img, ctx.feature_masks, sam2_ckpt, device)
        for name, mask_arr in ctx.feature_masks.items():
            cv2.imwrite(os.path.join(ctx.mask_dir, f"{name.lower()}.png"), mask_arr)

        ctx.alpha_dirs = {}
        for feature in FEATURES:
            feature_out = os.path.join(ctx.pipeline_workspace, f"matte_{feature.lower()}")
            os.makedirs(feature_out, exist_ok=True)
            video_name = os.path.basename(ctx.render_in)
            ctx.alpha_dirs[feature] = os.path.join(feature_out, video_name, "pha")

        print("🎬 Running SAM2 tracking...")
        from sam2.build_sam import build_sam2_video_predictor

        sam2_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml", sam2_ckpt, device=device
        )

        print("  🌀 SAM2: initializing tracking memory...")
        inference_state = sam2_predictor.init_state(video_path=ctx.render_in)
        sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=1, mask=ctx.feature_masks["Lips"])
        sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=2, mask=ctx.feature_masks["Skin"])
        sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=3, mask=ctx.feature_masks["Eyes"])

        for feature in FEATURES:
            os.makedirs(ctx.alpha_dirs[feature], exist_ok=True)

        print("  Propagating...")
        obj_id_to_feature = {1: "Lips", 2: "Skin", 3: "Eyes"}
        frames_written = {f: 0 for f in FEATURES}
        propagation_loop = sam2_predictor.propagate_in_video(inference_state)

        for frame_idx, object_ids, mask_logits in propagation_loop:
            actual_frame = ctx.frame_mapper.to_sequence(frame_idx)
            for i, obj_id in enumerate(object_ids):
                feature = obj_id_to_feature.get(obj_id)
                if not feature:
                    continue
                raw = mask_logits[i]
                matte = ((raw > 0.0).cpu().numpy() * 255).astype(np.uint8)
                matte = matte.squeeze()
                if matte.ndim != 2:
                    print(f"    ⚠️ {feature} frame {actual_frame}: unexpected shape {matte.shape}, skipping")
                    continue
                out_path = os.path.join(ctx.alpha_dirs[feature], f"{actual_frame:05d}.png")
                ok = cv2.imwrite(out_path, matte, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok:
                    print(f"    ❌ cv2.imwrite FAILED: {out_path} (matte shape={matte.shape}, dtype={matte.dtype})")
                elif frame_idx == 0:
                    print(f"    {feature} frame {actual_frame}: shape={matte.shape}, nonzero={np.count_nonzero(matte)}, wrote={out_path}")
                frames_written[feature] += 1

        for feat, count in frames_written.items():
            print(f"    {feat}: {count} frames written to {ctx.alpha_dirs[feat]}")
        print("  ✓ SAM2 tracking complete")

        if ctx.args.mode == "color":
            self._combine_alphas(ctx)

        print("✓ All matte sequences committed to disk.")
        return ctx

    def _combine_alphas(self, ctx: PipelineContext):
        print("  Combining alphas into multicolor canvas...")
        combined_dir = ctx.matte_dirs["Combined"]
        ref_list = sorted(f for f in os.listdir(ctx.alpha_dirs["Skin"]) if f.endswith(".png"))
        color_map = {
            "Lips": np.array([0, 0, 255], dtype=np.uint8),   # BGR red
            "Skin": np.array([255, 0, 0], dtype=np.uint8),   # BGR blue
            "Eyes": np.array([0, 255, 0], dtype=np.uint8),   # BGR green
        }
        combine_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        for fname in ref_list:
            canvas = np.zeros((ctx.h, ctx.w, 3), dtype=np.uint8)
            for feat, color in color_map.items():
                a = cv2.imread(os.path.join(ctx.alpha_dirs[feat], fname), cv2.IMREAD_GRAYSCALE)
                if a is not None:
                    a3 = (a.astype(np.float32) / 255.0)[:, :, np.newaxis]
                    canvas = cv2.add(canvas, (color.reshape(1, 1, 3) * a3).astype(np.uint8))
            combine_pool.submit(cv2.imwrite, os.path.join(combined_dir, fname), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        combine_pool.shutdown(wait=True)


class ConformStage:
    """Stage 5: Import matte sequences into Scratch."""

    def run(self, ctx: PipelineContext) -> PipelineContext:
        print(f"🎨 Importing mattes as {ctx.args.target.upper()}...")

        targets_to_load = FEATURES if ctx.args.mode == "grayscale" else ["Combined"]
        matte_frame_tc = ctx.frame_mapper.to_timecode()

        for feature in targets_to_load:
            media_path = ctx.alpha_dirs.get(feature, ctx.matte_dirs.get(feature))
            label = f"AI_Matte_{feature}" if ctx.args.mode == "grayscale" else "AI_Clown_Pass_Matte"
            naming_suffix = f"{ctx.shot_name}_{feature}"

            if ctx.args.target == "layer":
                result = ctx.scratch.add_matte_layer(
                    ctx.shot_uuid, label, media_path, slip=ctx.frame_mapper.slip_offset,
                    source_props={
                        "reel_id": ctx.source_reel,
                        "fps": getattr(ctx.shot_data, "fps", None),
                        "frame_tc": matte_frame_tc,
                        "name": naming_suffix,
                    }
                )
                if result and hasattr(result, "name"):
                    print(f"  ✓ Layer '{label}' created on shot '{ctx.shot_name}'")
                else:
                    print(f"  ⚠️ Failed to create layer '{label}'")
            else:
                result = ctx.scratch.create_shot(label, media_path)
                if result and hasattr(result, "uuid"):
                    new_uuid = str(result.uuid)
                    print(f"  ✓ Shot '{label}' created [UUID: {new_uuid}]")
                    ctx.scratch.set_shot_properties(
                        new_uuid,
                        reel_id=ctx.source_reel,
                        fps=getattr(ctx.shot_data, "fps", None),
                        frame_tc=matte_frame_tc,
                        name=naming_suffix
                    )
                else:
                    print(f"  ⚠️ Failed to create shot '{label}'")
        return ctx


class CleanupStage:
    """Stage 6: Cache cleanup and shot notes."""

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.args.auto_clean:
            print("🧹 [--auto-clean] Removing rendered cache...")
            if os.path.exists(ctx.render_in):
                shutil.rmtree(ctx.render_in)
                print(f"  ✓ Purged: {ctx.render_in}")

        note_text = (
            f"AI Face Matte Pipeline Completed.\n"
            f"Mode: {ctx.args.mode.upper()} | Target: {ctx.args.target.upper()}\n"
            f"Source Timecode: {ctx.source_tc} | Reel: {ctx.source_reel}\n"
            f"Frames: {ctx.frame_mapper.slip_offset} + {ctx.shot_length} frames"
        )
        try:
            ctx.scratch.append_note(ctx.shot_uuid, note_text, frame=ctx.frame_mapper.slip_offset)
            print("📝 Pipeline note appended to shot.")
        except Exception as e:
            print(f"⚠️ Could not append note: {e}")

        print(f"🚀 Pipeline complete. AI mattes loaded as {ctx.args.target.upper()} on '{ctx.shot_name}'.")
        return ctx


# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================
class Pipeline:
    """Orchestrates the 6 pipeline stages in sequence."""

    def __init__(self, args: argparse.Namespace, scratch: ScratchAPI):
        self._stages = [
            MetadataStage(),
            RenderStage(),
            SegFaceStage(),
            SAM2Stage(),
            ConformStage(),
            CleanupStage(),
        ]
        self._ctx = PipelineContext(args=args, scratch=scratch)

    def run(self) -> PipelineContext:
        for stage in self._stages:
            self._ctx = stage.run(self._ctx)
        return self._ctx


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="AI Face Matte Pipeline for Assimilate Scratch")
    # Scratch UI passes: -P1 <mode> -P2 <cleanup> -P3 <target> -P4 <model>
    parser.add_argument("-P1", type=int, choices=[0, 1], required=True,
                        help="Output mode: 0=grayscale (per-feature mattes), 1=color (combined multicolor)")
    parser.add_argument("-P2", type=str, choices=["y", "n"], required=True,
                        help="Auto-clean rendered cache: y=yes, n=no")
    parser.add_argument("-P3", type=int, choices=[0, 1], default=0,
                        help="Timeline destination: 0=layer, 1=version")
    parser.add_argument("-P4", type=int, choices=[0, 1, 2, 3], default=0,
                        help="SegFace model: 0=ConvNeXt-CelebA, 1=SwinB-CelebA, 2=SwinV2B-CelebA, 3=SwinB-LaPa")
    args = parser.parse_args()

    # Map Scratch UI parameters to internal names used throughout the pipeline
    args.mode = "color" if args.P1 == 1 else "grayscale"
    args.auto_clean = (args.P2 == "y")
    args.target = "version" if args.P3 == 1 else "layer"
    segface_models = ["segface_convnext", "segface_swinb", "segface_swinv2b", "segface_swinb_lapa"]
    args.segface_model = segface_models[args.P4]

    return args


def main():
    args = parse_args()
    scratch = ScratchAPI()
    print(f"🛰️ API Bridge Engaged | Mode: {args.mode.upper()} | Target: {args.target.upper()} | Model: {args.segface_model}")
    Pipeline(args, scratch).run()


if __name__ == "__main__":
    main()
