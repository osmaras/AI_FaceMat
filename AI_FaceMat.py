#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python>=4.8.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "torch==2.5.1+cu124",
#     "torchvision==0.20.1+cu124",
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
from pathlib import Path
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
    "segface": {
        "repo_id": "kartiknarayan/SegFace",
        "filename": "convnext_celeba_512/model_299.pt",
        "backbone": "segface_celeb",
        "model": "convnext_base",
    },
    "sam2": {
        "repo_id": "facebook/sam2.1-hiera-large",
        "filename": "sam2.1_hiera_large.pt",
    },
}

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

disk_writer_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

def async_write_image(file_path, image_data, is_png=True):
    """Offloads file operations cleanly to background CPU cores."""
    if is_png:
        cv2.imwrite(file_path, image_data, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    else:
        cv2.imwrite(file_path, image_data)

# ==============================================================================
# RUNTIME ENVIRONMENT PIPELINE
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="AI Face Matte Pipeline for Assimilate Scratch")
    # Scratch UI passes: -P1 <mode> -P2 <cleanup> -P3 <target>
    parser.add_argument("-P1", type=int, choices=[0, 1], required=True,
                        help="Output mode: 0=grayscale (per-feature mattes), 1=color (combined multicolor)")
    parser.add_argument("-P2", type=str, choices=["y", "n"], required=True,
                        help="Auto-clean rendered cache: y=yes, n=no")
    parser.add_argument("-P3", type=int, choices=[0, 1], default=0,
                        help="Timeline destination: 0=layer, 1=version")
    args = parser.parse_args()

    # Map Scratch UI parameters to internal names used throughout the pipeline
    args.mode = "color" if args.P1 == 1 else "grayscale"
    args.auto_clean = (args.P2 == "y")
    args.target = "version" if args.P3 == 1 else "layer"

    return args

def main():
    args = parse_args()
    scratch = ScratchAPI()
    print(f"🛰️ API Bridge Engaged | Mode: {args.mode.upper()} | Target: {args.target.upper()}")

    # ----------------------------------------------------------------------
    # STAGE 1: METADATA & DATA PATH EVALUATION
    # API: GET /constructs/current/sel_shots?level=ALL
    #      GET /projects/current
    # ----------------------------------------------------------------------
    selection, shot_data = scratch.get_selected_shot()
    if not selection or not shot_data:
        print("❌ No shot selected in the current construct. Aborting.")
        return

    shot_uuid = str(selection.uuid)
    shot_name = shot_data.name or "VFX_Shot"
    frame_tc = shot_data.frame_tc or 0

    # Get the TIMELINE slot length (not the shot media length)
    slot_idx = int(getattr(selection, "slot_idx", 0) or 0)
    slot_data = scratch.get_slot(slot_idx)
    slot_length = int(getattr(slot_data, "length", 0) or 0)
    shot_length = slot_length if slot_length > 0 else (shot_data.length or 1)
    print(f"  ↳ Slot {slot_idx} length: {slot_length} | Shot media length: {shot_data.length}")

    # Determine start frame: prefer handle in_point, fall back to frame_tc
    if shot_data.handles and shot_data.handles.frame_in is not None:
        in_point = shot_data.handles.frame_in
    else:
        in_point = frame_tc

    # Source conform metadata (reel_id, timecode live on ShotData directly)
    source_tc = str(shot_data.timecode) if shot_data.timecode else "00:00:00:00"
    source_reel = shot_data.reel_id or "A001C001"

    # Resolve workspace from project cache path, media path, or shot file location
    proj_paths = scratch.get_project_paths()
    if proj_paths and proj_paths.cache_path:
        workspace_base = proj_paths.cache_path
    elif proj_paths and proj_paths.media_path:
        workspace_base = proj_paths.media_path
    elif shot_data.file:
        workspace_base = os.path.dirname(shot_data.file)
    else:
        workspace_base = os.getcwd()

    PIPELINE_WORKSPACE = os.path.join(workspace_base, "Cache", "AI_Pipeline_Workspace", shot_uuid)
    RENDER_IN = os.path.join(PIPELINE_WORKSPACE, "source_sequence")
    os.makedirs(RENDER_IN, exist_ok=True)

    FEATURES = ["Skin", "Lips", "Eyes"]
    matte_dirs = {}

    if args.mode == "grayscale":
        for feature in FEATURES:
            matte_dirs[feature] = os.path.join(PIPELINE_WORKSPACE, f"matte_{feature.lower()}")
            os.makedirs(matte_dirs[feature], exist_ok=True)
    else:
        matte_dirs["Combined"] = os.path.join(PIPELINE_WORKSPACE, "matte_multicolor")
        os.makedirs(matte_dirs["Combined"], exist_ok=True)

    print(f"🎬 Active Shot: '{shot_name}' [UUID: {shot_uuid}]")
    print(f"📂 Workspace: {PIPELINE_WORKSPACE}")
    print(f"🎞️ Frame Range: {in_point} + {shot_length} frames")

    # ----------------------------------------------------------------------
    # STAGE 2: RENDER FRAMES TO DISK VIA SNAPSHOT API
    # Uses shot length directly — no output node needed.
    # ----------------------------------------------------------------------
    existing_frames = [f for f in os.listdir(RENDER_IN) if f.endswith('.png')]
    if len(existing_frames) >= shot_length:
        print(f"🎞️ Found {len(existing_frames)} cached frames, skipping render pass.")
    else:
        print(f"🎞️ Rendering {shot_length} frames via snapshot API...")
        for i in range(shot_length):
            frame_num = in_point + i
            fpath = os.path.join(RENDER_IN, f"{frame_num:05d}.png")
            scratch.render_frame_snapshot(shot_uuid, frame_num, fpath)
            if (i + 1) % 10 == 0 or (i + 1) == shot_length:
                print(f"\r  ↳ Rendered {i + 1}/{shot_length} frames", end="", flush=True)
        print()

    frame_list = sorted([f for f in os.listdir(RENDER_IN) if f.endswith('.png')])
    if not frame_list:
        print("❌ Error: Render cache folder is empty. Cannot continue.")
        return

    # ----------------------------------------------------------------------
    # STAGE 3: AI ANALYSIS & THREAD-POOLED DISK WRITE
    # ----------------------------------------------------------------------
    print("🧠 Initializing deep learning models on GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-resolve checkpoints (HF cache hit = instant, miss = download once)
    segface_spec = CHECKPOINTS["segface"]
    segface_ckpt = ensure_checkpoint("segface")
    face_parser = SegFaceParser(
        checkpoint=segface_ckpt,
        device=device,
        backbone=segface_spec["backbone"],
        model_name=segface_spec["model"],
    )

    sample_img = cv2.imread(os.path.join(RENDER_IN, frame_list[0]))
    h, w, _ = sample_img.shape

    # --- Keyframe Seeding: SegFace semantic parsing on frame 0 ---
    # CelebAMask-HQ indices: 2=skin, 8=l_eye, 9=r_eye, 12=l_lip, 13=u_lip
    print("🎭 Running SegFace semantic parsing on frame 0...")
    parsed_semantic_map = face_parser.parse_image(sample_img)

    # Save per-feature binary masks as PNGs
    mask_dir = os.path.join(PIPELINE_WORKSPACE, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    feature_masks = {
        "Skin": (parsed_semantic_map == 2).astype(np.uint8) * 255,
        "Lips": np.isin(parsed_semantic_map, [12, 13]).astype(np.uint8) * 255,
        "Eyes": np.isin(parsed_semantic_map, [8, 9]).astype(np.uint8) * 255,
    }
    for name, mask_arr in feature_masks.items():
        cv2.imwrite(os.path.join(mask_dir, f"{name.lower()}.png"), mask_arr)
        print(f"    {name}: {np.count_nonzero(mask_arr)} nonzero / {mask_arr.size} total pixels")

    # --- SAM2 Image Predictor: refine mask boundaries on frame 0 ---
    # Uses box prompts from SegFace + distance-field edge boosting for thin features.
    # Only runs on the keyframe — the video tracker handles temporal propagation.
    sam2_ckpt = ensure_checkpoint("sam2")
    feature_masks = refine_masks_sam2_image(sample_img, feature_masks, sam2_ckpt, device)
    # Overwrite refined masks on disk
    for name, mask_arr in feature_masks.items():
        cv2.imwrite(os.path.join(mask_dir, f"{name.lower()}.png"), mask_arr)

    # Prepare matte output dirs and alpha_dirs
    alpha_dirs = {}
    for feature in FEATURES:
        feature_out = os.path.join(PIPELINE_WORKSPACE, f"matte_{feature.lower()}")
        os.makedirs(feature_out, exist_ok=True)
        video_name = os.path.basename(RENDER_IN)
        alpha_dirs[feature] = os.path.join(feature_out, video_name, "pha")

    # =====================================================================
    # SAM2 VIDEO TRACKING
    # =====================================================================
    print("🎬 Running SAM2 tracking...")

    # Lazy import SAM2 video predictor
    from sam2.build_sam import build_sam2_video_predictor

    sam2_predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", sam2_ckpt, device=device
    )

    # SAM2 requires JPEG frames — convert PNGs to a temp JPG directory
    sam2_frames_dir = os.path.join(PIPELINE_WORKSPACE, "sam2_frames")
    os.makedirs(sam2_frames_dir, exist_ok=True)
    for png_name in frame_list:
        jpg_name = os.path.splitext(png_name)[0] + ".jpg"
        jpg_path = os.path.join(sam2_frames_dir, jpg_name)
        if not os.path.exists(jpg_path):
            img = cv2.imread(os.path.join(RENDER_IN, png_name))
            cv2.imwrite(jpg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # SAM2: init state and register per-feature masks on frame 0
    print("  🌀 SAM2: initializing tracking memory...")
    inference_state = sam2_predictor.init_state(video_path=sam2_frames_dir)
    # obj_id mapping: 1=Lips, 2=Skin, 3=Eyes
    sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=1, mask=feature_masks["Lips"])
    sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=2, mask=feature_masks["Skin"])
    sam2_predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=3, mask=feature_masks["Eyes"])

    # Prepare output dirs for alpha frames
    for feature in FEATURES:
        os.makedirs(alpha_dirs[feature], exist_ok=True)

    # SAM2 propagation → write binary masks directly as mattes
    print("  Propagating...")
    obj_id_to_feature = {1: "Lips", 2: "Skin", 3: "Eyes"}
    frames_written = {f: 0 for f in FEATURES}
    propagation_loop = sam2_predictor.propagate_in_video(inference_state)

    for frame_idx, object_ids, mask_logits in propagation_loop:
        actual_frame = in_point + frame_idx
        for i, obj_id in enumerate(object_ids):
            feature = obj_id_to_feature.get(obj_id)
            if not feature:
                continue
            raw = mask_logits[i]
            matte = ((raw > 0.0).cpu().numpy() * 255).astype(np.uint8)
            # Ensure 2D (H, W) — squeeze any extra dims from SAM2
            matte = matte.squeeze()
            if matte.ndim != 2:
                print(f"    ⚠️ {feature} frame {actual_frame}: unexpected shape {matte.shape}, skipping")
                continue
            out_path = os.path.join(alpha_dirs[feature], f"{actual_frame:05d}.png")
            ok = cv2.imwrite(out_path, matte, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                print(f"    ❌ cv2.imwrite FAILED: {out_path} (matte shape={matte.shape}, dtype={matte.dtype})")
            elif frame_idx == 0:
                print(f"    {feature} frame {actual_frame}: shape={matte.shape}, nonzero={np.count_nonzero(matte)}, wrote={out_path}")
            frames_written[feature] += 1

    for feat, count in frames_written.items():
        print(f"    {feat}: {count} frames written to {alpha_dirs[feat]}")
    print("  ✓ SAM2 tracking complete")

    # --- Color mode: combine per-feature alphas into multicolor canvas ---
    if args.mode == "color":
        print("  Combining alphas into multicolor canvas...")
        combined_dir = matte_dirs["Combined"]
        ref_list = sorted(f for f in os.listdir(alpha_dirs["Skin"]) if f.endswith(".png"))
        color_map = {
            "Lips": np.array([0, 0, 255], dtype=np.uint8),   # BGR red
            "Skin": np.array([255, 0, 0], dtype=np.uint8),   # BGR blue
            "Eyes": np.array([0, 255, 0], dtype=np.uint8),   # BGR green
        }
        combine_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        for fname in ref_list:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            for feat, color in color_map.items():
                a = cv2.imread(os.path.join(alpha_dirs[feat], fname), cv2.IMREAD_GRAYSCALE)
                if a is not None:
                    a3 = (a.astype(np.float32) / 255.0)[:, :, np.newaxis]
                    canvas = cv2.add(canvas, (color.reshape(1, 1, 3) * a3).astype(np.uint8))
            combine_pool.submit(async_write_image, os.path.join(combined_dir, fname), canvas)
        combine_pool.shutdown(wait=True)

    print("✓ All matte sequences committed to disk.")

    # ----------------------------------------------------------------------
    # STAGE 4: CONFORMANCE — IMPORT MATTES INTO SCRATCH
    # Layer mode:  POST /shot/{shot_uuid}/layers/new  (LayerData with MatteData)
    # Version mode: POST /shot/new                    (new ShotData from sequence)
    # Then:         PUT  /shot/{shot_uuid}             (copy source reel_id/name)
    # ----------------------------------------------------------------------
    print(f"🎨 Importing mattes as {args.target.upper()}...")

    targets_to_load = FEATURES if args.mode == "grayscale" else ["Combined"]

    for feature in targets_to_load:
        # Grayscale: per-feature alphas from MatAnyone2 output
        # Color: combined multicolor canvas
        media_path = alpha_dirs.get(feature, matte_dirs.get(feature))
        label = f"AI_Matte_{feature}" if args.mode == "grayscale" else "AI_Clown_Pass_Matte"
        naming_suffix = f"{shot_name}_{feature}"

        # Matte timecode = shot start TC + in_point offset (trimmed start)
        matte_frame_tc = frame_tc + in_point

        if args.target == "layer":
            # Add as a matte layer on the existing shot, slipped to align with shot frame range
            result = scratch.add_matte_layer(
                shot_uuid, label, media_path, slip=in_point,
                source_props={
                    "reel_id": source_reel,
                    "fps": getattr(shot_data, "fps", None),
                    "frame_tc": matte_frame_tc,
                    "name": naming_suffix,
                }
            )
            if result and hasattr(result, "name"):
                print(f"  ✓ Layer '{label}' created on shot '{shot_name}'")
            else:
                print(f"  ⚠️ Failed to create layer '{label}'")
        else:
            # Create a new shot entry from the matte sequence
            result = scratch.create_shot(label, media_path)
            if result and hasattr(result, "uuid"):
                new_uuid = str(result.uuid)
                print(f"  ✓ Shot '{label}' created [UUID: {new_uuid}]")
                # Conform source metadata onto the new shot
                scratch.set_shot_properties(
                    new_uuid,
                    reel_id=source_reel,
                    fps=getattr(shot_data, "fps", None),
                    frame_tc=matte_frame_tc,
                    name=naming_suffix
                )
            else:
                print(f"  ⚠️ Failed to create shot '{label}'")

    # ----------------------------------------------------------------------
    # STAGE 5: CLEANUP & NOTIFICATIONS
    # PUT /shot/{shot_uuid}  (append note)
    # NOTE: No UI toast API exists in the Scratch REST API; use print output.
    # ----------------------------------------------------------------------
    if args.auto_clean:
        print("🧹 [--auto-clean] Removing rendered cache...")
        if os.path.exists(RENDER_IN):
            shutil.rmtree(RENDER_IN)
            print(f"  ✓ Purged: {RENDER_IN}")
        sam2_tmp = os.path.join(PIPELINE_WORKSPACE, "sam2_frames")
        if os.path.exists(sam2_tmp):
            shutil.rmtree(sam2_tmp)
            print(f"  ✓ Purged: {sam2_tmp}")

    note_text = (
        f"AI Face Matte Pipeline Completed.\n"
        f"Mode: {args.mode.upper()} | Target: {args.target.upper()}\n"
        f"Source Timecode: {source_tc} | Reel: {source_reel}\n"
        f"Frames: {in_point} + {shot_length} frames"
    )
    try:
        scratch.append_note(shot_uuid, note_text, frame=in_point)
        print("📝 Pipeline note appended to shot.")
    except Exception as e:
        print(f"⚠️ Could not append note: {e}")

    print(f"🚀 Pipeline complete. AI mattes loaded as {args.target.upper()} on '{shot_name}'.")

if __name__ == "__main__":
    main()
