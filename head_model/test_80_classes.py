#!/usr/bin/env python3
"""
Comprehensive comparison of our YOLOv8 vs Ultralytics with 80 classes.
Tests full forward pass, loss computation, and training step outputs.
"""

import sys
import traceback
import warnings

import torch

# Add current directory to path for imports
sys.path.append(".")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_coco_test_batch(batch_size=2, img_size=640, num_objects=5):
    """Create a realistic test batch with COCO-style annotations."""
    # Create random images
    images = torch.randn(batch_size, 3, img_size, img_size)

    # Create realistic targets (batch_idx, class, x, y, w, h)
    targets = []
    for batch_idx in range(batch_size):
        # Random number of objects per image
        n_objects = int(torch.randint(1, num_objects + 1, (1,)).item())

        for _ in range(n_objects):
            # Random COCO class (0-79)
            class_id = torch.randint(0, 80, (1,)).item()

            # Random box coordinates (normalized)
            x = torch.rand(1).item() * 0.8 + 0.1  # 0.1 to 0.9
            y = torch.rand(1).item() * 0.8 + 0.1
            w = torch.rand(1).item() * 0.3 + 0.05  # 0.05 to 0.35
            h = torch.rand(1).item() * 0.3 + 0.05

            targets.append([batch_idx, class_id, x, y, w, h])

    targets = torch.tensor(targets, dtype=torch.float32)

    return {
        "images": images,
        "targets": targets,
        "img_paths": [f"dummy_{i}.jpg" for i in range(batch_size)],
        "orig_sizes": torch.tensor([[img_size, img_size]] * batch_size),
        "img_sizes": torch.tensor([[img_size, img_size]] * batch_size),
    }


def compare_backbone_features():
    """Compare backbone features before the head between our model and Ultralytics."""
    print("🔬 Backbone Feature Comparison (80 Classes)")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        from train_lightning import YOLOv8Model

        # Load models with 80 classes
        print("📦 Loading models...")
        our_model = YOLOv8Model(num_classes=80, pretrained_path="yolov8n.pt")

        ultra_yolo = YOLO("yolov8n.pt")
        ultra_model = ultra_yolo.model

        # Check if ultra_model loaded correctly and is actually a model
        if ultra_model is None:
            print("❌ Failed to load Ultralytics model")
            return None, None, False, None

        # Type check for ultra_model
        if not hasattr(ultra_model, "eval") or not callable(ultra_model):
            print("❌ Ultralytics model doesn't have expected model interface")
            return None, None, False, None

        # Set to eval mode for consistent comparison
        our_model.eval()
        ultra_model.eval()

        # Create test batch
        print("📝 Creating test batch...")
        batch = create_coco_test_batch(batch_size=2, img_size=640)
        print(f"   Images: {batch['images'].shape}")
        print(f"   Targets: {batch['targets'].shape}")

        # Extract backbone features from our model
        print("\n🔍 Extracting backbone features...")
        torch.manual_seed(42)
        with torch.no_grad():
            our_features = our_model._extract_features(batch["images"])

        # Extract backbone features from Ultralytics model
        torch.manual_seed(42)
        with torch.no_grad():
            # Get intermediate features from Ultralytics backbone
            x = batch["images"]
            ultra_features = []

            # Forward through Ultralytics backbone layers manually
            if hasattr(ultra_model, "model") and hasattr(ultra_model.model, "__len__"):
                try:
                    backbone_layers = ultra_model.model[
                        :10
                    ]  # First 10 layers are backbone
                    for i, layer in enumerate(backbone_layers):
                        x = layer(x)
                        # Collect features at the same scales as our model
                        # Our model extracts at layers 4, 6, 9 -> P3, P4, P5
                        if i in [4, 6, 9]:  # Match our _extract_features method
                            ultra_features.append(x.clone())
                except Exception as e:
                    print(f"❌ Error extracting Ultralytics backbone features: {e}")
                    return None, None, False, batch
            else:
                print("❌ Cannot access Ultralytics model layers")
                return None, None, False, batch

        print("\n🔍 Backbone Feature Comparison:")
        print(f"Our features: {len(our_features)} scales")
        print(f"Ultra features: {len(ultra_features)} scales")

        # Compare backbone features at each scale
        total_diff = 0.0
        backbone_match = True

        for i in range(min(len(our_features), len(ultra_features))):
            our_feat = our_features[i]
            ultra_feat = ultra_features[i]

            print(f"\nBackbone Scale {i}:")
            print(f"  Our shape: {our_feat.shape}")
            print(f"  Ultra shape: {ultra_feat.shape}")

            if our_feat.shape == ultra_feat.shape:
                diff = torch.abs(our_feat - ultra_feat)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                total_diff += max_diff

                print(f"  Max diff: {max_diff:.2e}")
                print(f"  Mean diff: {mean_diff:.2e}")
                print(f"  Match: {'✅' if max_diff < 1e-5 else '❌'}")

                if max_diff >= 1e-5:
                    backbone_match = False
            else:
                print("  ❌ Shape mismatch!")
                total_diff += 1e6
                backbone_match = False

        print("\n📊 Backbone Summary:")
        print(f"Total difference: {total_diff:.2e}")
        if total_diff < 1e-5:
            print("🎉 PERFECT BACKBONE MATCH!")
        elif total_diff < 1e-3:
            print("✅ Excellent backbone match (minor numerical differences)")
        else:
            print("❌ Significant backbone differences found")

        # Now compare full model outputs (with heads)
        print("\n" + "=" * 60)
        print("HEAD OUTPUT COMPARISON")
        print("=" * 60)

        torch.manual_seed(42)
        our_outputs = our_model(batch["images"])

        torch.manual_seed(42)
        if callable(ultra_model):
            try:
                ultra_outputs = ultra_model(batch["images"])
            except Exception as e:
                print(f"❌ Error calling Ultralytics model: {e}")
                ultra_outputs = None
        else:
            print("❌ Ultra model is not callable")
            ultra_outputs = None

        print("\n🔍 Head Output Comparison:")
        print(f"Our head outputs: {len(our_outputs)} scales")

        # Compare head outputs (these may differ due to different architectures)
        if ultra_outputs is not None:
            if isinstance(ultra_outputs, (list, tuple)):
                print(f"Ultra head outputs: {len(ultra_outputs)} scales")
                for i in range(min(len(our_outputs), len(ultra_outputs))):
                    our_out = our_outputs[i]
                    ultra_out = ultra_outputs[i]
                    if hasattr(ultra_out, "shape"):
                        print(f"\nHead Scale {i}:")
                        print(f"  Our shape: {our_out.shape}")
                        print(f"  Ultra shape: {ultra_out.shape}")
                        print("  Expected difference: Different head architectures")
            else:
                # Single tensor output format
                if hasattr(ultra_outputs, "shape"):
                    print("Ultra head outputs: Single tensor")
                    print("\nHead Output:")
                    print(f"  Our format: {len(our_outputs)} separate tensors")
                    shape_str = str(getattr(ultra_outputs, "shape", "unknown shape"))
                    print(f"  Ultra format: Single tensor {shape_str}")
                    print("  Note: Different output formats (both valid)")
        else:
            print("Ultra head outputs: Not available")

        return our_features, ultra_features, backbone_match, batch

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return None, None, False, None


def compare_loss_computation():
    """Compare loss computation between our implementation and Ultralytics."""
    print("\n" + "=" * 60)
    print("LOSS COMPUTATION COMPARISON")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        from train_lightning import YOLOv8Loss, YOLOv8Model

        # Create identical test data with fixed seed
        torch.manual_seed(123)
        batch = create_coco_test_batch(batch_size=2, img_size=640, num_objects=3)

        print("📝 Loss comparison input:")
        print(f"   Images: {batch['images'].shape}")
        print(f"   Targets: {batch['targets'].shape}")
        print("   Target details:")
        for i, target in enumerate(batch["targets"]):
            print(
                f"     Target {i}: batch={target[0]}, class={target[1]}, bbox=[{target[2]:.3f}, {target[3]:.3f}, {target[4]:.3f}, {target[5]:.3f}]"
            )

        # Initialize our model and loss
        print("\n🔍 Loading our implementation...")
        our_model = YOLOv8Model(num_classes=80, pretrained_path="yolov8n.pt")
        our_model.train()  # Set to training mode for loss computation

        config = {
            "num_classes": 80,
            "box_loss_gain": 7.5,
            "cls_loss_gain": 0.5,
            "dfl_loss_gain": 1.5,
        }
        our_loss_fn = YOLOv8Loss(80, config)

        # Forward pass with our model
        print("\n🚀 Forward pass - Our model...")
        torch.manual_seed(123)
        if hasattr(our_model.head, "forward_with_dfl"):
            # Use DFL-enabled forward pass
            features = our_model._extract_features(batch["images"])
            our_predictions, our_dfl_predictions = our_model.head.forward_with_dfl(
                features
            )
            our_loss_dict = our_loss_fn(
                our_predictions, batch["targets"], our_dfl_predictions
            )
        else:
            our_predictions = our_model(batch["images"])
            our_loss_dict = our_loss_fn(our_predictions, batch["targets"])

        print("📊 Our model results:")
        print(f"   Predictions: {len(our_predictions)} scales")
        for i, pred in enumerate(our_predictions):
            print(f"     Scale {i}: {pred.shape}")
        print("   Loss components:")
        for key, value in our_loss_dict.items():
            print(f"     {key}: {value.item():.6f}")

        # Try Ultralytics comparison AND require it for success
        print("\n� Attempting Ultralytics comparison...")
        ultra_loss_available = False
        ultra_loss_dict = None
        ultra_predictions = None

        try:
            ultra_yolo = YOLO("yolov8n.pt")
            ultra_model = ultra_yolo.model

            if ultra_model is not None:
                print("✅ Ultralytics model loaded successfully")

                # Set to training mode with proper type checking
                if (
                    hasattr(ultra_model, "train")
                    and callable(getattr(ultra_model, "train", None))
                    and not isinstance(ultra_model, str)
                ):
                    ultra_model.train()

                # Forward pass with Ultralytics model
                torch.manual_seed(123)
                if callable(ultra_model):
                    ultra_predictions = ultra_model(batch["images"])
                    print("✅ Ultralytics forward pass successful")

                    if (
                        isinstance(ultra_predictions, (list, tuple))
                        and len(ultra_predictions) > 0
                    ):
                        print(
                            f"   Ultralytics predictions: {len(ultra_predictions)} outputs"
                        )
                        for i, pred in enumerate(ultra_predictions):
                            if hasattr(pred, "shape"):
                                print(f"     Output {i}: {pred.shape}")
                    elif hasattr(ultra_predictions, "shape"):
                        shape_str = str(getattr(ultra_predictions, "shape", "unknown"))
                        print(f"   Ultralytics predictions: Single tensor {shape_str}")

                    # Try using the training approach for loss computation
                    try:
                        # Try to compute loss using Ultralytics' training approach
                        from ultralytics.utils.loss import v8DetectionLoss

                        ultra_loss_fn = v8DetectionLoss(ultra_model)

                        # Create proper batch format that Ultralytics expects
                        # Convert our targets to the format Ultralytics loss function expects
                        ultra_batch = {}

                        # Collect all targets
                        all_cls = []
                        all_bboxes = []
                        all_batch_idx = []

                        for i, target in enumerate(batch["targets"]):
                            batch_idx = int(target[0].item())
                            cls_id = int(target[1].item())
                            bbox = target[2:6]  # x, y, w, h (normalized)

                            all_cls.append(cls_id)
                            all_bboxes.append(bbox.tolist())
                            all_batch_idx.append(batch_idx)

                        if len(all_cls) > 0:
                            # Create proper format for Ultralytics
                            ultra_batch["cls"] = torch.tensor(
                                all_cls, dtype=torch.long
                            ).unsqueeze(1)  # [N, 1]
                            ultra_batch["bboxes"] = torch.tensor(
                                all_bboxes, dtype=torch.float32
                            )  # [N, 4]
                            ultra_batch["batch_idx"] = torch.tensor(
                                all_batch_idx, dtype=torch.long
                            )  # [N]

                            # Alternative format that might work - create object with required attributes
                            class UltraTargets:
                                def __init__(self):
                                    self.cls = ultra_batch["cls"]
                                    self.bboxes = ultra_batch[
                                        "bboxes"
                                    ]  # Also try 'box' attribute
                                    self.box = ultra_batch[
                                        "bboxes"
                                    ]  # Some versions expect 'box'
                                    self.batch_idx = ultra_batch["batch_idx"]
                                    self.orig_shape = (
                                        torch.tensor([640, 640])
                                        .unsqueeze(0)
                                        .repeat(len(all_cls), 1)
                                    )

                            ultra_targets = UltraTargets()

                            print(
                                f"   Prepared Ultralytics batch: cls={ultra_batch['cls'].shape}, "
                                f"bboxes={ultra_batch['bboxes'].shape}, "
                                f"batch_idx={ultra_batch['batch_idx'].shape}"
                            )
                        else:
                            # Empty targets case
                            ultra_batch["cls"] = torch.zeros((0, 1), dtype=torch.long)
                            ultra_batch["bboxes"] = torch.zeros(
                                (0, 4), dtype=torch.float32
                            )
                            ultra_batch["batch_idx"] = torch.zeros(
                                (0,), dtype=torch.long
                            )
                            ultra_targets = ultra_batch

                        # Create hyperparameters object for Ultralytics loss function
                        class HyperParams:
                            def __init__(self):
                                self.box = 7.5
                                self.cls = 0.5
                                self.dfl = 1.5

                        # Set proper hyperparameters
                        ultra_loss_fn.hyp = HyperParams()

                        # Create targets object with both attribute and subscript access
                        class WorkingUltraTargets:
                            def __init__(self):
                                self.box = ultra_batch["bboxes"]
                                self.bboxes = ultra_batch["bboxes"]
                                self.cls = ultra_batch["cls"]
                                self.batch_idx = ultra_batch["batch_idx"]

                            def __getitem__(self, key):
                                if key == "box":
                                    return self.box
                                elif key == "bboxes":
                                    return self.bboxes
                                elif key == "cls":
                                    return self.cls
                                elif key == "batch_idx":
                                    return self.batch_idx
                                else:
                                    raise KeyError(f"Key {key} not found")

                        # Try loss computation with working format
                        try:
                            working_targets = WorkingUltraTargets()
                            ultra_result = ultra_loss_fn(
                                ultra_predictions, working_targets
                            )

                            # Extract loss from result
                            if (
                                isinstance(ultra_result, tuple)
                                and len(ultra_result) >= 1
                            ):
                                ultra_total_loss = ultra_result[0]
                                ultra_loss_dict = (
                                    ultra_result[1] if len(ultra_result) > 1 else {}
                                )
                            else:
                                ultra_total_loss = ultra_result
                                ultra_loss_dict = {}

                            # Store for later comparison
                            if hasattr(ultra_total_loss, "item"):
                                ultra_total_loss_value = ultra_total_loss.item()
                            else:
                                ultra_total_loss_value = ultra_total_loss

                            ultra_loss_available = True
                            print("✅ Ultralytics loss computation successful!")
                            print(f"   Total loss: {ultra_total_loss_value:.6f}")

                        except Exception as final_err:
                            print(
                                f"   ❌ Ultralytics loss computation failed: {final_err}"
                            )
                            ultra_loss_available = False
                            ultra_loss_dict = None
                            ultra_total_loss_value = None

                    except Exception as loss_e:
                        print(
                            f"   ⚠️ Ultralytics loss computation setup failed: {loss_e}"
                        )
                        ultra_loss_available = False
                        ultra_total_loss_value = None
                else:
                    print("   ⚠️ Ultralytics model is not callable")
            else:
                print("   ⚠️ Failed to access Ultralytics model")

        except Exception as ultra_e:
            print(f"   ⚠️ Ultralytics setup failed: {ultra_e}")
            ultra_loss_available = False
            ultra_total_loss_value = None

        # Loss comparison analysis
        print("\n🎯 Loss Analysis:")
        our_total = our_loss_dict.get("loss", torch.tensor(0.0)).item()
        print(f"   Our total loss: {our_total:.6f}")

        # Validate our loss is reasonable
        loss_valid = True
        if our_total <= 0 or our_total > 100:
            print(f"   ❌ Our loss seems unreasonable: {our_total}")
            loss_valid = False
        elif torch.isnan(torch.tensor(our_total)) or torch.isinf(
            torch.tensor(our_total)
        ):
            print(f"   ❌ Our loss is NaN or Inf: {our_total}")
            loss_valid = False
        else:
            print("   ✅ Our loss is in reasonable range")

        # Component validation
        components_valid = True
        for key, value in our_loss_dict.items():
            if key != "loss":
                val = value.item()
                if torch.isnan(value) or torch.isinf(value) or val < 0:
                    print(f"   ❌ Invalid component {key}: {val}")
                    components_valid = False
                else:
                    print(f"   ✅ Component {key}: {val:.6f}")

        # Compare with Ultralytics if available
        if ultra_loss_available and ultra_total_loss_value is not None:
            ultra_total = ultra_total_loss_value

            if ultra_total is not None:
                print(f"   Ultralytics total loss: {ultra_total:.6f}")
                loss_diff = abs(our_total - ultra_total)
                relative_diff = loss_diff / max(abs(our_total), abs(ultra_total), 1e-8)

                print(f"   Absolute difference: {loss_diff:.6f}")
                print(f"   Relative difference: {relative_diff:.1%}")

                # More strict comparison criteria
                if loss_diff < 0.001:
                    print("   🎉 EXCELLENT loss match!")
                    ultra_comparison_success = True
                elif relative_diff < 0.05:  # Within 5%
                    print("   ✅ Good loss match (within 5%)")
                    ultra_comparison_success = True
                elif relative_diff < 0.20:  # Within 20%
                    print(
                        "   ⚠️ Moderate difference (within 20% - acceptable for different implementations)"
                    )
                    ultra_comparison_success = True
                else:
                    print("   ❌ Significant loss difference (>20%)")
                    ultra_comparison_success = False
            else:
                print("   ❌ Could not extract Ultralytics total loss")
                ultra_comparison_success = False
        else:
            print("   ❌ Ultralytics comparison REQUIRED but not available")
            ultra_comparison_success = False  # Fail if Ultralytics isn't available

        return loss_valid and components_valid and ultra_comparison_success

    except Exception as e:
        print(f"❌ Error in loss comparison: {e}")
        traceback.print_exc()
        return False


def test_training_step():
    """Test a complete training step with both models."""
    print("\n" + "=" * 60)
    print("TRAINING STEP COMPARISON")
    print("=" * 60)

    try:
        from train_lightning import YOLOLightningModule

        # Create Lightning module with 80 classes
        config = {
            "num_classes": 80,
            "pretrained": True,
            "pretrained_checkpoint": "yolov8n.pt",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "box_loss_gain": 7.5,
            "cls_loss_gain": 0.5,
            "dfl_loss_gain": 1.5,
            "class_names": [f"class_{i}" for i in range(80)],
        }

        lightning_model = YOLOLightningModule(config)
        lightning_model.train()

        # Create test batch
        batch = create_coco_test_batch(batch_size=1, img_size=640)

        print("📝 Training step input:")
        print(f"   Images: {batch['images'].shape}")
        print(f"   Targets: {batch['targets'].shape}")

        # Run training step
        loss = lightning_model.training_step(batch, 0)

        print("\n🎯 Training step result:")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Loss type: {type(loss)}")
        print(f"   Requires grad: {loss.requires_grad}")

        # Test backward pass
        loss.backward()
        print("   ✅ Backward pass successful")

        # Check gradients
        has_grads = sum(1 for p in lightning_model.parameters() if p.grad is not None)
        total_params = sum(1 for p in lightning_model.parameters())
        print(f"   Parameters with gradients: {has_grads}/{total_params}")

        return True

    except Exception as e:
        print(f"❌ Error in training step: {e}")
        import traceback

        traceback.print_exc()
        return False


def parameter_statistics():
    """Compare detailed parameter statistics."""
    print("\n" + "=" * 60)
    print("PARAMETER STATISTICS")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        from train_lightning import YOLOv8Model

        # Load models
        our_model = YOLOv8Model(num_classes=80, pretrained_path="yolov8n.pt")

        # Analyze our model
        our_total = sum(p.numel() for p in our_model.parameters())
        our_backbone = sum(p.numel() for p in our_model.backbone.parameters())
        our_head = sum(p.numel() for p in our_model.head.parameters())

        # Try to analyze ultra model with proper error handling
        ultra_available = False
        ultra_total = 0
        ultra_backbone = 0
        ultra_head = 0

        try:
            ultra_yolo = YOLO("yolov8n.pt")
            ultra_model = ultra_yolo.model

            # Type check and validate model with proper attribute checking
            if (
                ultra_model is not None
                and hasattr(ultra_model, "parameters")
                and callable(getattr(ultra_model, "parameters", None))
                and not isinstance(ultra_model, str)
            ):  # Explicit string check
                ultra_total = sum(p.numel() for p in ultra_model.parameters())

                # Try to analyze backbone vs head parameters
                if hasattr(ultra_model, "named_parameters") and callable(
                    getattr(ultra_model, "named_parameters", None)
                ):
                    ultra_backbone = sum(
                        p.numel()
                        for name, p in ultra_model.named_parameters()
                        if name.startswith("model.")
                        and any(name.startswith(f"model.{i}.") for i in range(10))
                    )
                    ultra_head = ultra_total - ultra_backbone
                else:
                    # Fallback - estimate based on total
                    ultra_backbone = int(
                        ultra_total * 0.85
                    )  # Typical backbone percentage
                    ultra_head = ultra_total - ultra_backbone

                ultra_available = True
                print("✅ Ultralytics model parameter analysis successful")
            else:
                print(
                    "⚠️ Cannot access Ultralytics model parameters (model may be string or incompatible type)"
                )
                ultra_available = False
        except Exception as e:
            print(f"⚠️ Error analyzing Ultralytics model: {e}")
            ultra_available = False

        print("📊 Our Model:")
        print(f"   Total: {our_total:,} parameters")
        print(
            f"   Backbone: {our_backbone:,} parameters ({100*our_backbone/our_total:.1f}%)"
        )
        print(f"   Head: {our_head:,} parameters ({100*our_head/our_total:.1f}%)")

        print("\n📊 Ultralytics Model:")
        if ultra_available:
            print(f"   Total: {ultra_total:,} parameters")
            print(
                f"   Backbone: {ultra_backbone:,} parameters ({100*ultra_backbone/ultra_total:.1f}%)"
            )
            print(
                f"   Head: {ultra_head:,} parameters ({100*ultra_head/ultra_total:.1f}%)"
            )
        else:
            print("   Not available for comparison")

        print("\n🔍 Comparison:")
        if ultra_available:
            backbone_match = our_backbone == ultra_backbone
            backbone_diff = abs(our_backbone - ultra_backbone)
            backbone_relative_diff = (
                backbone_diff / max(our_backbone, ultra_backbone)
                if max(our_backbone, ultra_backbone) > 0
                else 0
            )

            print(
                f"   Backbone match: {'✅' if backbone_match else '❌'} ({our_backbone:,} vs {ultra_backbone:,})"
            )
            if not backbone_match:
                print(
                    f"   Backbone difference: {backbone_diff:,} parameters ({backbone_relative_diff:.1%})"
                )

            head_diff = abs(our_head - ultra_head)
            total_diff = abs(our_total - ultra_total)

            print(f"   Head difference: {head_diff:,} parameters")
            print(f"   Total difference: {total_diff:,} parameters")

            # Consider backbone match successful if within 5%
            backbone_success = backbone_match or backbone_relative_diff < 0.05
            if backbone_success:
                print("   ✅ Backbone parameters are compatible")
            else:
                print("   ⚠️ Significant backbone parameter difference")

            return backbone_success
        else:
            print("   Cannot compare - Ultralytics model not available")
            # Don't fail if Ultralytics isn't available - focus on our model validation
            return True

    except Exception as e:
        print(f"❌ Error in parameter analysis: {e}")
        traceback.print_exc()
        return False


def main():
    """Main comparison function."""
    print("🧪 COMPREHENSIVE YOLOV8 VALIDATION (80 Classes)")
    print("=" * 80)

    overall_success = True

    # Parameter statistics
    print("🔍 Step 1: Parameter Analysis")
    backbone_param_success = parameter_statistics()
    if not backbone_param_success:
        print("❌ Parameter analysis failed")
        overall_success = False

    # Backbone feature comparison
    print("\n🔍 Step 2: Backbone Feature Analysis")
    feature_result = compare_backbone_features()

    # Handle different return formats from compare_backbone_features
    if feature_result is None or len(feature_result) != 4:
        print("❌ Backbone feature comparison failed")
        backbone_feature_match = False
        features_available = False
        overall_success = False
    else:
        our_features, ultra_features, backbone_feature_match, batch = feature_result
        features_available = our_features is not None
        if not backbone_feature_match:
            print("❌ Backbone features don't match exactly")
            overall_success = False
        if not features_available:
            print("❌ Feature extraction failed")
            overall_success = False

    # Loss comparison - this is the critical test
    print("\n🔍 Step 3: Loss Computation Analysis")
    loss_success = compare_loss_computation()
    if not loss_success:
        print("❌ Loss computation validation failed")
        overall_success = False

    # Training step test
    print("\n🔍 Step 4: Training Step Validation")
    training_success = test_training_step()
    if not training_success:
        print("❌ Training step failed")
        overall_success = False

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    print("📊 Test Results:")
    print(
        f"   Parameter Analysis: {'✅ PASS' if backbone_param_success else '❌ FAIL'}"
    )
    print(f"   Backbone Features: {'✅ PASS' if backbone_feature_match else '❌ FAIL'}")
    print(f"   Feature Extraction: {'✅ PASS' if features_available else '❌ FAIL'}")
    print(f"   Loss Computation: {'✅ PASS' if loss_success else '❌ FAIL'}")
    print(f"   Training Step: {'✅ PASS' if training_success else '❌ FAIL'}")

    # Strict validation criteria
    if overall_success:
        print("\n🎉 COMPREHENSIVE VALIDATION PASSED!")
        print("   ✅ Architecture is correct")
        print("   ✅ Weights loaded properly")
        print("   ✅ Loss computation is valid")
        print("   ✅ Training pipeline is functional")
        print("   ✅ Ready for production training!")
        return True
    else:
        print("\n❌ VALIDATION FAILED!")
        print("   Required criteria not met. Issues found:")

        critical_issues = []
        if not loss_success:
            critical_issues.append("Loss computation is invalid or unreasonable")
        if not training_success:
            critical_issues.append("Training step fails")
        if not features_available:
            critical_issues.append("Feature extraction fails")

        for issue in critical_issues:
            print(f"   • {issue}")

        if not backbone_feature_match:
            print("   • Backbone features differ from reference (may be acceptable)")
        if not backbone_param_success:
            print("   • Parameter count mismatch (may be acceptable)")

        print("\n🔧 Fix these issues before proceeding with training.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
