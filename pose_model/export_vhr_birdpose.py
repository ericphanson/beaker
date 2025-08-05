#!/usr/bin/env python3
"""
export_vhr_birdpose.py
----------------------
End-to-end utility for exporting the "S-add" flavour of VHR-BirdPose to an
optimised ONNX graph.

Usage
-----
uv run python export_vhr_birdpose.py
       --workdir ./checkpoints
       --config  ./w32_256x256_adam_lr1e-3_ak_vhr_s.yaml
       --opset   13
"""

import argparse
import hashlib
import os
import sys
import yaml
from pathlib import Path
import torch
from upstream.pose_vhr import get_pose_net

# Add the upstream directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "upstream"))

# ----------------------------------------------------------------------
# 1. CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--workdir",
    type=Path,
    default=Path("./checkpoints"),
    help="Where to save ONNX files",
)
parser.add_argument(
    "--config",
    type=Path,
    default=Path("./w32_256x256_adam_lr1e-3_ak_vhr_s.yaml"),
    help="Path to the YAML config for the S-add model",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("./vhr_birdpose_s_add.pth"),
    help="Path to the .pth checkpoint file",
)
parser.add_argument(
    "--opset", type=int, default=13, help="ONNX opset version (≥13 recommended)"
)
parser.add_argument(
    "--aggressive-quantization",
    action="store_true",
    help="Enable additional aggressive quantization techniques",
)
parser.add_argument(
    "--fp16", action="store_true", help="Export FP16 model in addition to INT8"
)
args = parser.parse_args()
args.workdir.mkdir(parents=True, exist_ok=True)

# Resolve paths
CKPT_PATH = Path(args.checkpoint).resolve()
CONFIG_PATH = Path(args.config).resolve()

if not CKPT_PATH.exists():
    print(f"✗ Checkpoint file not found: {CKPT_PATH}")
    sys.exit(1)

if not CONFIG_PATH.exists():
    print(f"✗ Config file not found: {CONFIG_PATH}")
    sys.exit(1)


# ----------------------------------------------------------------------
# 2. SHA-256 checksum
# ----------------------------------------------------------------------
def sha256sum(fp: Path, buf_size: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


ckpt_hash = sha256sum(CKPT_PATH)
print(f"✓ Checkpoint found. SHA-256 = {ckpt_hash}")


# ----------------------------------------------------------------------
# 3. Load configuration and model
# ----------------------------------------------------------------------
def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model_checkpoint(checkpoint_path, cfg):
    """Load the VHR-BirdPose model and its checkpoint."""

    print(
        f"Creating VHR-BirdPose model with '{cfg['MODEL']['EXTRA']['FUSE_STREGY']}' fusion strategy..."
    )

    # Create the model
    model = get_pose_net(cfg, is_train=False)

    # Load the checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("Found 'state_dict' key in checkpoint")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("Found 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint directly as state_dict")

    # Load the state dict
    try:
        # Apply key mapping for fit_out -> att_fit_out
        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("fit_out."):
                # Map fit_out.* to att_fit_out.*
                new_key = key.replace("fit_out.", "att_fit_out.")
                mapped_state_dict[new_key] = value
                print(f"Mapped key: {key} -> {new_key}")
            else:
                mapped_state_dict[key] = value

        missing_keys, unexpected_keys = model.load_state_dict(
            mapped_state_dict, strict=False
        )
        print("✓ Checkpoint loaded successfully!")
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}):")
            for key in missing_keys:
                print(f"  - {key}")
        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys:
                print(f"  + {key}")
    except Exception as e:
        print(f"⚠ Warning: Could not load some parameters: {e}")
        # Try to load with prefix removal if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove common prefixes if present
            new_key = k.replace("module.", "").replace("backbone.", "")
            # Also apply fit_out -> att_fit_out mapping
            if new_key.startswith("fit_out."):
                new_key = new_key.replace("fit_out.", "att_fit_out.")
            new_state_dict[new_key] = v

        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                new_state_dict, strict=False
            )
            print("✓ Checkpoint loaded successfully with key adjustments!")
        except Exception as e2:
            print(f"✗ Error loading checkpoint: {e2}")
            sys.exit(1)

    return model


print("Loading configuration...")
cfg = load_config_from_yaml(CONFIG_PATH)
print(f"✓ Configuration loaded from: {CONFIG_PATH}")

print("Loading model...")
model = load_model_checkpoint(CKPT_PATH, cfg)
model.eval()

# ----------------------------------------------------------------------
# 4. Export to ONNX
# ----------------------------------------------------------------------

# Use the input size from configuration
height, width = cfg["MODEL"]["IMAGE_SIZE"]
dummy = torch.randn(1, 3, height, width)  # dynamic shapes set below

ONNX_RAW = args.workdir / "vhr_birdpose_s_add_raw.onnx"
print(f"Exporting raw ONNX to {ONNX_RAW} …")

try:
    torch.onnx.export(
        model,
        dummy,
        ONNX_RAW.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["heatmaps"],
        dynamic_axes={
            "image": {0: "B", 2: "H", 3: "W"},
            "heatmaps": {0: "B", 2: "H_out", 3: "W_out"},
        },
    )
    print(f"✓ Raw ONNX exported successfully to {ONNX_RAW}")
except Exception as e:
    if "onnx is not installed" in str(e):
        print("✗ Error: ONNX is required for export but not installed.")
        print("  Install ONNX dependencies with:")
        print("  uv add onnx onnx-simplifier onnxruntime")
        sys.exit(1)
    else:
        print(f"✗ Error during ONNX export: {e}")
        sys.exit(1)

# ----------------------------------------------------------------------
# 5. Simplify (onnx-sim)
# ----------------------------------------------------------------------
print("Running onnx-simplifier …")
try:
    import onnx
    from onnxsim import simplify

    model_onnx = onnx.load(ONNX_RAW.as_posix())
    model_simp, check = simplify(model_onnx, dynamic_input_shape=True)
    assert check, "Simplified ONNX model could not be validated"
    ONNX_SIM = args.workdir / "vhr_birdpose_s_add_sim.onnx"
    onnx.save(model_simp, ONNX_SIM)
    print(f"✓ Simplified ONNX saved to {ONNX_SIM}")
except ImportError as e:
    print(f"⚠ Warning: Could not import onnx/onnxsim: {e}")
    print(
        "  Skipping simplification step. Install with: pip install onnx onnx-simplifier"
    )
    ONNX_SIM = ONNX_RAW

# ----------------------------------------------------------------------
# 6. Graph optimisation (ONNX Runtime)
# ----------------------------------------------------------------------
print("Applying ONNX Runtime graph optimiser …")
try:
    import onnxruntime as ort

    ONNX_OPT = args.workdir / "vhr_birdpose_s_add_opt.onnx"

    # Set up session options for optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    sess_options.optimized_model_filepath = str(ONNX_OPT)

    # Create session to trigger optimization and save optimized model
    print("  Creating inference session to apply optimizations...")
    session = ort.InferenceSession(str(ONNX_SIM), sess_options)

    # Verify the optimized model was created
    if ONNX_OPT.exists():
        print(f"✓ Optimized ONNX saved to {ONNX_OPT}")
    else:
        print(
            "⚠ Warning: Optimized model was not saved, using simplified model instead."
        )
        ONNX_OPT = ONNX_SIM

except ImportError as e:
    print(f"⚠ Warning: Could not import onnxruntime: {e}")
    print("  Skipping optimization step. Using simplified model instead.")
    ONNX_OPT = ONNX_SIM
except Exception as e:
    print(f"⚠ Warning: ONNX optimization failed: {e}")
    print("  Using simplified model instead.")
    ONNX_OPT = ONNX_SIM

# ----------------------------------------------------------------------
# 7. Preprocessing for quantization (optional but recommended)
# ----------------------------------------------------------------------
print("Preprocessing model for quantization …")
try:
    from onnxruntime.quantization.preprocess import quant_pre_process

    ONNX_PREPROC = args.workdir / "vhr_birdpose_s_add_preproc.onnx"

    # Try preprocessing with symbolic shape inference first
    try:
        quant_pre_process(
            input_model_path=str(ONNX_OPT),
            output_model_path=str(ONNX_PREPROC),
            skip_optimization=False,  # We want optimization
            skip_onnx_shape=False,  # We want shape inference
            skip_symbolic_shape=False,  # We want symbolic shape inference (good for transformers)
        )
        print(
            f"✓ Preprocessed model with symbolic shape inference saved to {ONNX_PREPROC}"
        )
    except Exception as e:
        print(f"⚠ Symbolic shape inference failed: {e}")
        print("  Trying preprocessing without symbolic shape inference...")

        # Fallback: try without symbolic shape inference
        quant_pre_process(
            input_model_path=str(ONNX_OPT),
            output_model_path=str(ONNX_PREPROC),
            skip_optimization=False,  # We want optimization
            skip_onnx_shape=False,  # We want shape inference
            skip_symbolic_shape=True,  # Skip symbolic shape inference
        )
        print(
            f"✓ Preprocessed model (without symbolic shape inference) saved to {ONNX_PREPROC}"
        )

    # Use preprocessed model for quantization
    quantization_input = ONNX_PREPROC

except ImportError as e:
    print(f"⚠ Warning: Could not import preprocessing: {e}")
    print("  Skipping preprocessing step. Using optimized model for quantization.")
    quantization_input = ONNX_OPT
except Exception as e:
    print(f"⚠ Warning: Preprocessing failed completely: {e}")
    print("  Using optimized model for quantization.")
    quantization_input = ONNX_OPT

# ----------------------------------------------------------------------
# 8. FP16 quantization (optional, faster alternative to INT8)
# ----------------------------------------------------------------------
if args.fp16:
    print("Creating FP16 model …")
    try:
        import onnx
        from onnxconverter_common import float16

        # Load the preprocessed model
        model_fp32 = onnx.load(str(quantization_input))

        # Convert to FP16
        model_fp16 = float16.convert_float_to_float16(model_fp32)

        ONNX_FP16 = args.workdir / "vhr_birdpose_s_add_fp16.onnx"
        onnx.save(model_fp16, str(ONNX_FP16))
        print(f"✓ FP16 model saved to {ONNX_FP16}")

    except ImportError as e:
        print(f"⚠ Warning: Could not import onnxconverter-common for FP16: {e}")
        print("  Install with: uv add onnxconverter-common")
    except Exception as e:
        print(f"⚠ Warning: FP16 conversion failed: {e}")

# ----------------------------------------------------------------------
# 9. Dynamic-range INT8 quantisation
# ----------------------------------------------------------------------
print("Quantising to dynamic-range INT8 …")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    ONNX_INT8 = args.workdir / "vhr_birdpose_s_add_int8.onnx"

    # Base quantization settings
    quant_kwargs = {
        "model_input": quantization_input.as_posix(),
        "model_output": ONNX_INT8.as_posix(),
        "weight_type": QuantType.QInt8,
    }

    # Add aggressive quantization options if requested
    if args.aggressive_quantization:
        print("  Using aggressive quantization settings...")
        # Quantize more operations
        quant_kwargs.update(
            {
                "nodes_to_quantize": None,  # Quantize all supported nodes
                "nodes_to_exclude": [],  # Don't exclude any nodes by default
            }
        )

        # Try to also quantize activations to INT8 (more aggressive)
        try:
            from onnxruntime.quantization import quantize_static, CalibrationDataReader

            print(
                "  Note: For best results with aggressive quantization, consider using static quantization"
            )
            print(
                "        This requires calibration data. Using dynamic quantization for now."
            )
        except ImportError:
            pass

    quantize_dynamic(**quant_kwargs)
    print(f"✓ Quantized ONNX saved to {ONNX_INT8}")

except ImportError as e:
    print(f"⚠ Warning: Could not import onnxruntime quantization: {e}")
    print("  Skipping quantization step.")
    ONNX_INT8 = quantization_input
except Exception as e:
    print(f"⚠ Warning: Quantization failed: {e}")
    print("  Using input model instead.")
    ONNX_INT8 = quantization_input

# ----------------------------------------------------------------------
# 10. Additional compression (optional)
# ----------------------------------------------------------------------
if args.aggressive_quantization:
    print("Applying additional model compression …")
    try:
        # Try to compress the quantized model further using ONNX tools
        import onnx
        from onnx.tools import optimizer

        # Load the quantized model
        model = onnx.load(str(ONNX_INT8))

        # Apply additional optimizations
        passes = [
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "extract_constant_to_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_consecutive_log_softmax",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
        ]

        print(f"  Applying {len(passes)} optimization passes...")
        optimized_model = optimizer.optimize(model, passes)

        ONNX_COMPRESSED = args.workdir / "vhr_birdpose_s_add_compressed.onnx"
        onnx.save(optimized_model, str(ONNX_COMPRESSED))
        print(f"✓ Compressed model saved to {ONNX_COMPRESSED}")

        # Update the final model reference
        ONNX_INT8 = ONNX_COMPRESSED

    except ImportError as e:
        print(f"⚠ Warning: Could not import ONNX optimizer: {e}")
        print("  Note: ONNX optimizer may not be available in all ONNX versions")
    except Exception as e:
        print(f"⚠ Warning: Additional compression failed: {e}")

print("✓ All done.")

# Show file sizes for comparison
import os


def format_size(bytes_size):
    """Format file size in human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


print("\nGenerated models:")
models_to_show = [
    ("Raw ONNX", ONNX_RAW),
    ("Simplified", ONNX_SIM) if ONNX_SIM != ONNX_RAW else None,
    ("Optimised", ONNX_OPT) if ONNX_OPT != ONNX_SIM else None,
    ("Preprocessed", ONNX_PREPROC)
    if "ONNX_PREPROC" in locals() and ONNX_PREPROC != ONNX_OPT
    else None,
]

# Add FP16 if created
if args.fp16 and "ONNX_FP16" in locals():
    models_to_show.append(("FP16", ONNX_FP16))

# Add final INT8 model
models_to_show.append(("INT8", ONNX_INT8))

for model_info in models_to_show:
    if model_info is not None:
        name, path = model_info
        if path.exists():
            size = os.path.getsize(path)
            print(f"  {name:12} : {path} ({format_size(size)})")

# Calculate compression ratio
if ONNX_RAW.exists() and ONNX_INT8.exists():
    original_size = os.path.getsize(ONNX_RAW)
    final_size = os.path.getsize(ONNX_INT8)
    compression_ratio = (1 - final_size / original_size) * 100
    print(
        f"\nCompression: {format_size(original_size)} → {format_size(final_size)} ({compression_ratio:.1f}% reduction)"
    )

if args.aggressive_quantization:
    print("\nFor even smaller models, consider:")
    print("  • Static quantization with calibration data")
    print("  • Model pruning (removing less important weights)")
    print("  • Knowledge distillation (training a smaller student model)")
    print("  • Architecture-specific optimizations")
