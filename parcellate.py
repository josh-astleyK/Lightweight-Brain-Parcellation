#!/usr/bin/env python3
"""
Whole-brain parcellation using trained UMamba models.

Produces a 134-region NIfTI parcellation from a T1-weighted MRI.
Input can be any resolution or orientation; the script resamples to 1 mm
isotropic RAS before inference and saves output in that resampled space.

Usage
-----
python parcellate.py --input t1.nii.gz --model umamba_lite
python parcellate.py --input t1.nii.gz --model umamba_base --device cuda
python parcellate.py --input scans/ --model umamba_fwr --output parcellations/

Models
------
  umamba_base      UMamba baseline  (~44 M params, highest accuracy)
  umamba_fwr       UMamba-FWR       (~11 M params, recommended)
  umamba_lite_int8 UMamba-lite INT8 (~11 M params, Linux x86 only)

Weight files are expected in models/weights/ by default.  Override with --weights-dir.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR / "models" / "weights"
PATCH = (160, 160, 160)
NUM_CLASSES = 134

# Maps CLI model name -> (architecture key, weight filename, precision)
MODELS = {
    "umamba_base":      ("base", "umamba_base.pth",      "fp32"),
    "umamba_fwr":       ("fwr",  "umamba_fwr.pth",       "fp32"),
    "umamba_lite_int8": ("lite", "umamba_lite_int8.pth", "int8"),
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image_path: str):
    """Load and preprocess a T1 image: RAS orientation, 1 mm iso, [0,1] intensity.

    Returns the preprocessed tensor, its MONAI metadata (contains the resampled
    affine needed to save the prediction), and the original nibabel image (used to
    resample the prediction back to the original image space on saving).
    """
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRangePercentilesd, EnsureTyped,
    )
    orig_img   = nib.load(image_path)   # keep original for resampling back
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])
    result = transforms({"image": image_path})
    tensor = result["image"].unsqueeze(0)   # add batch dim -> (1, 1, D, H, W)
    meta   = result["image"].meta
    return tensor, meta, orig_img


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def postprocess_largest_cc(pred: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of all labelled voxels."""
    from scipy.ndimage import label as ndi_label
    binary = (pred > 0).astype(np.uint8)
    labeled, n_cc = ndi_label(binary)
    if n_cc <= 1:
        return pred
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    brain_mask = labeled == int(np.argmax(sizes))
    out = pred.copy()
    out[~brain_mask] = 0
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _patch_mamba_for_cpu(model):
    """Replace CUDA selective-scan kernels with pure-Python reference ops."""
    import mamba_ssm.modules.mamba_simple as ms
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    try:
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref
    except Exception:
        causal_conv1d_ref = None
    ms.selective_scan_fn = selective_scan_ref
    ms.causal_conv1d_fn  = causal_conv1d_ref
    ms.mamba_inner_fn    = None
    for m in model.modules():
        if m.__class__.__name__ == "Mamba":
            m.use_fast_path = False
    return model


def load_model(model_name: str, weights_dir: Path, device: torch.device):
    from models.architecture import build_base, build_fwr, build_lite, MambaLayer

    arch, weight_file, precision = MODELS[model_name]
    weight_path = weights_dir / weight_file
    if not weight_path.exists():
        sys.exit(
            f"[error] weights not found: {weight_path}\n"
            f"        Download from the releases page and place in {weights_dir}/"
        )

    builders = {"base": build_base, "fwr": build_fwr, "lite": build_lite}
    model = builders[arch](num_classes=NUM_CLASSES, patch_size=PATCH).eval()

    if precision == "fp32":
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            state_dict = state_dict.get("model_state_dict", state_dict.get("model", state_dict))
        model.load_state_dict(state_dict)
        if device.type == "cpu":
            _patch_mamba_for_cpu(model)
        return model.to(device)

    # INT8 path — uses PyTorch static quantization (fbgemm backend, Linux x86 only).
    # The saved weights contain pre-quantised INT8 tensors; we rebuild the quantised
    # graph structure first, then load the weights into it.
    if torch.backends.quantized.engine not in ("fbgemm",):
        sys.exit(
            "[error] INT8 inference requires the fbgemm quantization backend "
            "(Linux x86).  Use --model umamba_fwr on this platform."
        )
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig

    torch.backends.quantized.engine = "fbgemm"
    # Exclude the Mamba bottleneck from quantisation (SSM ops are not supported)
    qconfig = get_default_qconfig_mapping("fbgemm").set_module_name("mamba_layer", None)
    custom_config = PrepareCustomConfig().set_non_traceable_module_classes([MambaLayer])
    prepared = prepare_fx(model, qconfig, example_inputs=(torch.randn(1, 1, *PATCH),),
                          prepare_custom_config=custom_config)
    quantised = convert_fx(prepared)
    quantised.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False))
    quantised.eval()
    if device.type == "cpu":
        _patch_mamba_for_cpu(quantised)
    return quantised.to(device)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, vol: torch.Tensor, device: torch.device) -> np.ndarray:
    """Sliding-window inference (160³ patch, 50% overlap, Gaussian blending)."""
    from monai.inferers import sliding_window_inference
    vol = vol.to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            vol, roi_size=PATCH, sw_batch_size=1,
            predictor=model, overlap=0.5, mode="gaussian",
        )
    return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int16)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_parcellation(pred: np.ndarray, meta, orig_img: nib.Nifti1Image, output_path: Path):
    """Resample prediction back to the original image space and save as NIfTI.

    Inference runs on a 1 mm isotropic RAS image; this resamples the integer
    label map back to the native resolution and orientation of the input scan
    using nearest-neighbour interpolation (correct for label maps).
    """
    from nibabel.processing import resample_from_to

    try:
        pred_affine = meta["affine"].numpy()
    except Exception:
        pred_affine = np.eye(4)

    pred_nii = nib.Nifti1Image(pred.astype(np.int16), pred_affine)
    output_nii = resample_from_to(pred_nii, (orig_img.shape[:3], orig_img.affine), order=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_nii, str(output_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_output_path(input_path: Path, output_arg, model_name: str) -> Path:
    if output_arg is None:
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        return input_path.parent / f"{stem}_{model_name}_parcellation.nii.gz"
    out = Path(output_arg)
    if out.is_dir() or (not out.suffix and not out.exists()):
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        return out / f"{stem}_{model_name}_parcellation.nii.gz"
    return out


def find_inputs(input_arg: str):
    p = Path(input_arg)
    if p.is_dir():
        files = sorted(p.glob("*.nii.gz")) + sorted(p.glob("*.nii"))
        if not files:
            sys.exit(f"[error] no .nii.gz / .nii files found in {p}")
        return files
    if not p.exists():
        sys.exit(f"[error] input not found: {p}")
    return [p]


def main():
    parser = argparse.ArgumentParser(
        description="Whole-brain parcellation using UMamba models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",       required=True,
                        help="T1 NIfTI file (.nii.gz) or directory of images")
    parser.add_argument("--model",       default="umamba_fwr",
                        choices=list(MODELS.keys()),
                        help="model variant (default: umamba_fwr)")
    parser.add_argument("--output",      default=None,
                        help="output path or directory (default: alongside input)")
    parser.add_argument("--device",      default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="inference device (default: auto)")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR),
                        help="directory containing .pth weight files (default: models/weights/)")
    parser.add_argument("--threads",     type=int, default=None,
                        help="CPU threads (default: PyTorch default)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    weights_dir = Path(args.weights_dir)
    inputs = find_inputs(args.input)

    print(f"Model   : {args.model}")
    print(f"Device  : {device}")
    print(f"Weights : {weights_dir}")
    print(f"Images  : {len(inputs)}")
    print()

    model = load_model(args.model, weights_dir, device)

    for img_path in inputs:
        t0 = time.perf_counter()
        print(f"Processing {img_path.name} ...", end=" ", flush=True)

        vol, meta, orig_img = preprocess(str(img_path))
        pred = run_inference(model, vol, device)
        pred = postprocess_largest_cc(pred)

        out_path = get_output_path(img_path, args.output, args.model)
        save_parcellation(pred, meta, orig_img, out_path)

        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s)  ->  {out_path}")

    print("\nRegion names: assets/label_map.json")


if __name__ == "__main__":
    main()
