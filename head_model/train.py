#!/usr/bin/env python3
# ruff: noqa: E402
"""
Train YOLOv8n model for bird head detection on M1 MacBook Pro with Comet.ml tracking.
"""

import os
import re
import glob
import tqdm.auto as _tqdm_auto

import comet_ml
import torch
import torch.nn as nn
from ultralytics import YOLO


# hacks for MPS especially for yolov12
def safe_view(t, *shape):
    return t.view(*shape) if t.is_contiguous() else t.reshape(*shape)


setattr(torch.Tensor, "safe_view", safe_view)  # type: ignore[attr-defined]
# save original implementation
_orig_bn_forward = nn.BatchNorm2d.forward


def _mps_safe_forward(self, input):
    # MPS requires contiguous input, CUDA/CPU ignore the copy flag
    if not input.is_contiguous():
        input = input.contiguous()
    return _orig_bn_forward(self, input)


# patch every existing and future BatchNorm2d
nn.BatchNorm2d.forward = _mps_safe_forward

# NMS prefilter patch will be applied later based on TRAINING_CONFIG

os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"  # C++ sites as well
torch.autograd.set_detect_anomaly(True)  # Python call chain

if torch.backends.cuda.is_built():
    torch.backends.cudnn.benchmark = True

# Training Configuration
TRAINING_CONFIG = {
    "model": "yolov8n",
    "model_file": "yolov8n.pt",
    "model_yaml": "yolov8n.yaml",
    "data": "../data/yolo-4-class/dataset.yaml",
    "epochs": 100,
    "imgsz": 960,
    "batch": 8,
    "project": "runs/multi-detect",
    "name": "bird_multi_yolov8n",
    "workers": 0,  # Prevent multiprocessing issues on M1
    "verbose": True,
    # Inference/eval knobs
    "conf": 0.10,
    "iou": 0.50,
    "max_nms": 12000,
    "max_det": 200,
    "plots": True,
    # Enable/disable Fast NMS prefilter monkey patch
    "fast_nms_prefilter": True,
    # Optional resumable training
    "resume": True,
    "exist_ok": False,
    # Per-epoch prediction logging
    "log_epoch_predictions": True,  # set True to enable
    "pred_samples": 8,  # number of val images to predict/log per epoch
    "pred_log_interval": 1,  # log every N epochs
    # Progress bar verbosity / ETA
    "progress_bar_eta": True,
    "tqdm_bar_format": "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    # Epoch sample logging to Comet
    "log_epoch_samples": True,
    "task": "bird_detection",
    "dataset": "CUB-200-2011",
    "architecture": "YOLOv8n",
    # Debug Configuration
    "debug_run": True,  # Set to True for quick testing
    "debug_epochs": 5,  # Reduced epochs for debug
    "debug_fraction": 0.1,  # Use 10% of data for debug (0.1 = 10%)
}

# if TRAINING_CONFIG["debug_run"]:
# print("DEBUG: Setting image size to 640 for debug run")
# TRAINING_CONFIG["imgsz"] = 640  # Smaller size for debug

# --- Conditionally apply Fast NMS prefilter (speeds up validation on MPS/CPU) ---
if TRAINING_CONFIG.get("fast_nms_prefilter", True):
    try:
        from ultralytics.utils import ops as _uops

        _orig_nms = _uops.non_max_suppression

        def _prefilter_topk(prediction, nc, max_k):
            # prediction: (B, N, 5+nc)
            if prediction is None or prediction.ndim != 3:
                return prediction
            outs = []
            for x in prediction:  # per image (N, 5+nc)
                if x.numel() == 0:
                    outs.append(x)
                    continue
                # combined score = obj * max_cls
                if nc > 0 and x.shape[1] >= 5 + nc:
                    scores = x[:, 4] * x[:, 5 : 5 + nc].amax(1)
                else:
                    scores = x[:, 4]
                K = min(int(max_k), x.shape[0])
                if K < x.shape[0]:
                    topk_idx = scores.topk(K).indices
                    x = x[topk_idx]
                outs.append(x)
            # pad to equal length if shapes differ before stacking
            maxN = max(t.shape[0] for t in outs) if outs else 0
            if maxN == 0:
                return prediction
            padded = []
            for t in outs:
                if t.shape[0] < maxN:
                    pad = torch.zeros(
                        (maxN - t.shape[0], t.shape[1]), device=t.device, dtype=t.dtype
                    )
                    t = torch.cat([t, pad], 0)
                padded.append(t)
            return torch.stack(padded, 0)

        def non_max_suppression_fast(
            prediction,
            conf_thres=0.001,
            iou_thres=0.6,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nc=0,
            max_time_img=2.5,
            max_nms=30000,
            max_wh=7680,
        ):
            # Prefilter to top‑K candidates per image to cut NMS workload dramatically
            try:
                K = min(12000, max_nms)
                prediction = _prefilter_topk(prediction, nc, K)
            except Exception:
                pass  # fall back silently
            return _orig_nms(
                prediction,
                conf_thres,
                iou_thres,
                classes,
                agnostic,
                multi_label,
                labels,
                max_det,
                nc,
                max_time_img,
                max_nms,
                max_wh,
            )

        _uops.non_max_suppression = non_max_suppression_fast
        print("⚡ Patched Ultralytics NMS with top‑K prefilter (K≤12k).")
    except Exception as _e:
        print(f"⚠️ NMS prefilter patch skipped: {_e}")
else:
    print("ℹ️ NMS prefilter patch disabled via TRAINING_CONFIG.")


def print_checkpoint_metrics(trainer):
    """Print trainer metrics and loss details after each checkpoint is saved."""
    print(
        f"Model details\n"
        f"Best fitness: {trainer.best_fitness}, "
        f"Loss names: {trainer.loss_names}, "  # List of loss names
        f"Metrics: {trainer.metrics}, "
        f"Total loss: {trainer.tloss}"  # Total loss value
    )


CALLBACKS = {
    "on_model_save": print_checkpoint_metrics,
}

# Global references for callbacks
COMET_EXPERIMENT = None
GLOBAL_YOLO = None


def on_train_start_log_comet_key(trainer):
    global COMET_EXPERIMENT
    try:
        if COMET_EXPERIMENT is not None:
            save_dir = str(trainer.save_dir) if hasattr(trainer, "save_dir") else None
            if save_dir:
                write_comet_key(save_dir, getattr(COMET_EXPERIMENT, "id", ""))
    except Exception as _e:
        print(f"⚠️ Could not persist Comet key at train start: {_e}")


# add callback
CALLBACKS["on_train_start"] = on_train_start_log_comet_key


# ----------------- Run directory helpers -----------------
from typing import Optional, Tuple


def find_last_run_with_checkpoint(
    project_dir: str, base_name: str
) -> Tuple[Optional[str], Optional[str]]:
    """Return (resolved_run_name, last_ckpt_path) for the latest numeric-suffixed run with last.pt, else (None, None)."""
    try:
        entries = [
            d
            for d in os.listdir(project_dir)
            if os.path.isdir(os.path.join(project_dir, d))
        ]
    except FileNotFoundError:
        return None, None
    pat = re.compile(rf"^{re.escape(base_name)}(\d+)?$")
    candidates = []  # (suffix_int, dir_name)
    for d in entries:
        m = pat.match(d)
        if m:
            suffix = int(m.group(1)) if m.group(1) else 1
            candidates.append((suffix, d))
    candidates.sort(reverse=True)
    for _, d in candidates:
        ckpt = os.path.join(project_dir, d, "weights", "last.pt")
        if os.path.exists(ckpt):
            return d, ckpt
    # also check base without suffix explicitly
    base_ckpt = os.path.join(project_dir, base_name, "weights", "last.pt")
    if os.path.exists(base_ckpt):
        return base_name, base_ckpt
    return None, None


def next_run_name(project_dir: str, base_name: str) -> str:
    """Return base_name if unused, else the next available numeric-suffixed name (base2, base3, ...)."""
    base_path = os.path.join(project_dir, base_name)
    if not os.path.exists(base_path):
        return base_name
    # find max suffix
    try:
        entries = [
            d
            for d in os.listdir(project_dir)
            if os.path.isdir(os.path.join(project_dir, d))
        ]
    except FileNotFoundError:
        return base_name
    pat = re.compile(rf"^{re.escape(base_name)}(\d+)?$")
    max_suffix = 1
    for d in entries:
        m = pat.match(d)
        if m:
            s = int(m.group(1)) if m.group(1) else 1
            if s > max_suffix:
                max_suffix = s
    return f"{base_name}{max_suffix + 1}"


def comet_key_file(run_dir: str) -> str:
    return os.path.join(run_dir, "comet_experiment_key.txt")


def read_comet_key(run_dir: str) -> Optional[str]:
    try:
        with open(comet_key_file(run_dir), "r") as f:
            return f.read().strip() or None
    except Exception:
        return None


def write_comet_key(run_dir: str, key: str) -> None:
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(comet_key_file(run_dir), "w") as f:
            f.write(key)
    except Exception as _e:
        print(f"⚠️ Failed to write Comet experiment key to {run_dir}: {_e}")


# ---------------------------------------------------------


def setup_comet(device, experiment_key: Optional[str] = None):
    """Setup Comet.ml experiment tracking. If experiment_key is provided, resume that experiment."""
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        print("⚠️  COMET_API_KEY not found. Comet.ml tracking will be disabled.")
        print("   Set your API key: export COMET_API_KEY='your-api-key'")
        print("   Get your API key from: https://www.comet.ml/api/my/settings/")
        return None

    project_name = os.getenv("COMET_PROJECT_NAME", "bird-head-detector")
    workspace = os.getenv("COMET_WORKSPACE")

    try:
        if experiment_key:
            experiment = comet_ml.ExistingExperiment(
                api_key=api_key,
                previous_experiment=experiment_key,
                project_name=project_name,
                workspace=workspace,
            )
            print(f"✅ Resumed Comet.ml experiment: {experiment.url}")
        else:
            experiment = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
            )
            print(f"✅ Comet.ml experiment started: {experiment.url}")

        # Log hyperparameters from global config
        log_params = TRAINING_CONFIG.copy()
        log_params["device"] = device
        experiment.log_parameters(log_params)
        return experiment

    except Exception as e:
        print(f"❌ Failed to initialize Comet.ml: {e}")
        print("   Training will continue without experiment tracking.")
        return None


def create_debug_dataset():
    """Create a subset dataset configuration for debug runs."""
    import random
    import shutil
    from pathlib import Path

    import yaml

    debug_dir = Path("../data/yolo-4-class-debug")

    # Remove existing debug directory to prevent accumulation of old files
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
        print(f"🗑️  Removed existing debug directory: {debug_dir}")

    debug_dir.mkdir(exist_ok=True)

    # Create debug directories
    for split in ["train", "val"]:
        (debug_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (debug_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy subset of files
    original_dir = Path("../data/yolo-4-class")
    fraction = TRAINING_CONFIG["debug_fraction"]

    for split in ["train", "val"]:
        original_images = list((original_dir / split / "images").glob("*.jpg"))
        sample_size = max(1, int(len(original_images) * fraction))
        sampled_images = random.sample(original_images, sample_size)

        print(
            f"📊 Debug: Using {len(sampled_images)}/{len(original_images)} {split} images ({fraction * 100:.1f}%)"
        )

        for img_path in sampled_images:
            # Copy image
            dst_img = debug_dir / split / "images" / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            src_label = original_dir / split / "labels" / label_name
            dst_label = debug_dir / split / "labels" / label_name
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)

    # Create debug dataset.yaml
    debug_yaml = {
        "path": str(debug_dir.absolute()),  # Use absolute path
        "train": "train/images",
        "val": "val/images",
        "nc": 4,
        "names": ["bird", "head", "eye", "beak"],
    }

    yaml_path = debug_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(debug_yaml, f, default_flow_style=False)

    return str(yaml_path)


def main():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available")
        device = "mps"  # Re-enable MPS for speed
        print("� Using MPS for accelerated training")
    else:
        print("❌ MPS not available, falling back to CPU")
        device = "cpu"

    # Check if running in debug mode
    is_debug = TRAINING_CONFIG["debug_run"]
    if is_debug:
        print("🐛 DEBUG MODE: Quick training run enabled")
        print(
            f"   - Epochs: {TRAINING_CONFIG['debug_epochs']} (vs {TRAINING_CONFIG['epochs']} normal)"
        )
        print(
            f"   - Data subset: {TRAINING_CONFIG['debug_fraction'] * 100:.1f}% of full dataset"
        )

        # Create debug dataset
        data_config = create_debug_dataset()
        epochs = TRAINING_CONFIG["debug_epochs"]
        run_name = f"{TRAINING_CONFIG['name']}_debug"
    else:
        print("🚀 FULL TRAINING MODE")
        data_config = TRAINING_CONFIG["data"]
        epochs = TRAINING_CONFIG["epochs"]
        run_name = TRAINING_CONFIG["name"]

    # Determine run_name and resume state earlier (supports numeric suffixes)
    project_dir = TRAINING_CONFIG["project"]
    resume_flag = bool(TRAINING_CONFIG.get("resume", False))
    last_ckpt = None
    experiment_key_for_resume = None
    if resume_flag:
        resolved, last = find_last_run_with_checkpoint(project_dir, run_name)
        if resolved and last:
            run_name = resolved
            last_ckpt = last
            run_dir = os.path.join(project_dir, run_name)
            experiment_key_for_resume = read_comet_key(run_dir)
            print(f"🔁 Resuming training from checkpoint: {last_ckpt}")
            if experiment_key_for_resume:
                print(
                    f"🔗 Found Comet experiment key in run dir: {experiment_key_for_resume}"
                )
            else:
                print(
                    "ℹ️  No Comet experiment key found; a new experiment will be created."
                )
        else:
            print(
                f"ℹ️  Resume requested but no checkpoint found under '{project_dir}' for base '{TRAINING_CONFIG['name']}'. Starting fresh."
            )
            resume_flag = False
    else:
        # choose the next available run name so we can predict the save directory
        run_name = next_run_name(project_dir, run_name)

    # Setup Comet.ml tracking with device info (resume with key if available)
    experiment = setup_comet(device, experiment_key=experiment_key_for_resume)
    global COMET_EXPERIMENT
    COMET_EXPERIMENT = experiment

    # Log debug/resume info to Comet.ml
    if experiment:
        experiment.log_parameter(
            "resume_requested", TRAINING_CONFIG.get("resume", False)
        )
        experiment.log_parameter("resume_active", resume_flag)
        if last_ckpt:
            experiment.log_parameter("resume_checkpoint", last_ckpt)
        experiment.log_parameter("run_name", run_name)
        experiment.log_parameter("debug_mode", is_debug)
        if is_debug:
            experiment.log_parameter("debug_epochs", epochs)
            experiment.log_parameter(
                "debug_fraction", TRAINING_CONFIG["debug_fraction"]
            )

    # Load a pretrained YOLO model
    if resume_flag and last_ckpt:
        print(f"📦 Loading checkpoint for resume: {last_ckpt}")
        model = YOLO(last_ckpt)
    else:
        print(f"📦 Loading {TRAINING_CONFIG['model']} pretrained model...")
        model = YOLO(TRAINING_CONFIG["model_yaml"]).load(TRAINING_CONFIG["model_file"])

    # Store model reference for callbacks
    global GLOBAL_YOLO
    GLOBAL_YOLO = model

    # Register callbacks
    for name, value in CALLBACKS.items():
        model.add_callback(name, value)

    # Configure Comet.ml integration for YOLO
    if experiment:
        os.environ["COMET_MODE"] = "online"
        # If resuming, ensure the key file exists in run dir now
        if resume_flag:
            try:
                run_dir = os.path.join(project_dir, run_name)
                if getattr(experiment, "id", None):
                    write_comet_key(run_dir, experiment.id)
            except Exception:
                pass

    # Train the model on bird head dataset
    mode_text = "DEBUG" if is_debug else "FULL"
    print(f"🚀 Starting {mode_text} training...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=TRAINING_CONFIG["imgsz"],
        batch=TRAINING_CONFIG["batch"],
        device=device,
        project=TRAINING_CONFIG["project"],
        name=run_name,
        workers=TRAINING_CONFIG["workers"],
        verbose=TRAINING_CONFIG["verbose"],
        conf=TRAINING_CONFIG["conf"],
        iou=TRAINING_CONFIG["iou"],
        max_det=TRAINING_CONFIG["max_det"],
        plots=TRAINING_CONFIG["plots"],
        resume=resume_flag,
        exist_ok=(True if resume_flag else TRAINING_CONFIG.get("exist_ok", False)),
        amp=True,
    )
    # Log final results to Comet.ml
    if experiment and results:
        try:
            # Log final metrics
            final_metrics = (
                results.results_dict if hasattr(results, "results_dict") else {}
            )
            for key, value in final_metrics.items():
                if isinstance(value, int | float):
                    experiment.log_metric(f"final_{key}", value)

            # Log model artifacts
            best_model_path = f"{TRAINING_CONFIG['project']}/{run_name}/weights/best.pt"
            if os.path.exists(best_model_path):
                experiment.log_model("best_model", best_model_path)
                print("📤 Model uploaded to Comet.ml")

            experiment.end()
            print("✅ Comet.ml experiment completed")

        except Exception as e:
            print(f"⚠️  Error logging to Comet.ml: {e}")

    mode_text = "DEBUG" if is_debug else "FULL"
    print(f"✅ {mode_text} training completed!")
    print(
        f"📊 Best model saved to: {TRAINING_CONFIG['project']}/{run_name}/weights/best.pt"
    )

    return results


# --- Conditionally patch tqdm to show ETA ---
if TRAINING_CONFIG.get("progress_bar_eta", True):
    try:
        from tqdm.auto import tqdm as _tqdm

        _orig_tqdm = _tqdm

        def _tqdm_with_eta(*args, **kwargs):
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("smoothing", 0.05)
            kwargs.setdefault(
                "bar_format",
                TRAINING_CONFIG.get(
                    "tqdm_bar_format",
                    "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ),
            )
            return _orig_tqdm(*args, **kwargs)

        _tqdm_auto.tqdm = _tqdm_with_eta  # type: ignore[attr-defined]
        print("📈 Patched tqdm progress bar to show ETA.")
    except Exception as _e:
        print(f"⚠️ Progress bar patch skipped: {_e}")


# --- Debugging and Per-Epoch Prediction Logging ---


def on_fit_epoch_end_log_preds(trainer):
    # Log epoch-level sample/batch info to Comet
    try:
        if COMET_EXPERIMENT is not None and TRAINING_CONFIG.get(
            "log_epoch_samples", True
        ):
            epoch = int(getattr(trainer, "epoch", -1))
            bs = getattr(trainer, "batch_size", None)
            nb = getattr(trainer, "nb", None)
            train_samples = None
            if bs is not None and nb is not None:
                train_samples = int(bs) * int(nb)
            else:
                tl = getattr(trainer, "train_loader", None)
                try:
                    if tl is not None and hasattr(tl, "dataset"):
                        train_samples = len(tl.dataset)
                except Exception:
                    pass
            val_dl = getattr(getattr(trainer, "validator", None), "dataloader", None)
            val_samples = None
            try:
                if val_dl is not None and hasattr(val_dl, "dataset"):
                    val_samples = len(val_dl.dataset)
            except Exception:
                pass
            if epoch >= 0:
                if train_samples is not None:
                    COMET_EXPERIMENT.log_metric(
                        "train_samples_per_epoch", int(train_samples), step=epoch
                    )
                if val_samples is not None:
                    COMET_EXPERIMENT.log_metric(
                        "val_samples_per_epoch", int(val_samples), step=epoch
                    )
                if nb is not None:
                    COMET_EXPERIMENT.log_metric(
                        "train_batches_per_epoch", int(nb), step=epoch
                    )
                if bs is not None:
                    COMET_EXPERIMENT.log_metric("batch_size", int(bs), step=epoch)
    except Exception as _e:
        print(f"⚠️ Failed to log epoch sample metrics: {_e}")

    if not TRAINING_CONFIG.get("log_epoch_predictions", False):
        return
    try:
        epoch = int(getattr(trainer, "epoch", -1))
        interval = max(1, int(TRAINING_CONFIG.get("pred_log_interval", 1)))
        if epoch >= 0 and (epoch % interval) != 0:
            return
        # Determine a small set of validation images
        ds = getattr(getattr(trainer, "validator", None), "dataloader", None)
        paths = []
        try:
            if (
                ds is not None
                and hasattr(ds, "dataset")
                and hasattr(ds.dataset, "im_files")
            ):
                paths = list(ds.dataset.im_files)[
                    : int(TRAINING_CONFIG.get("pred_samples", 8))
                ]
        except Exception:
            paths = []
        if not paths or GLOBAL_YOLO is None:
            return
        # Save predictions under the train run directory
        project_dir = str(trainer.save_dir)
        pred_project = os.path.join(project_dir, "epoch_preds")
        pred_name = f"epoch_{epoch:03d}"
        # Run prediction and save images
        GLOBAL_YOLO.predict(
            source=paths,
            imgsz=TRAINING_CONFIG["imgsz"],
            conf=TRAINING_CONFIG["conf"],
            iou=TRAINING_CONFIG["iou"],
            save=True,
            project=pred_project,
            name=pred_name,
            verbose=False,
        )
        # Log to Comet
        if COMET_EXPERIMENT is not None:
            out_dir = os.path.join(pred_project, pred_name)
            for img_path in sorted(
                glob.glob(os.path.join(out_dir, "*.jpg"))
                + glob.glob(os.path.join(out_dir, "*.png"))
            ):
                try:
                    COMET_EXPERIMENT.log_image(img_path, step=epoch, image_format="png")
                except Exception:
                    pass
    except Exception as _e:
        print(f"⚠️ Failed to save/log epoch predictions: {_e}")


CALLBACKS["on_fit_epoch_end"] = on_fit_epoch_end_log_preds


if __name__ == "__main__":
    main()
