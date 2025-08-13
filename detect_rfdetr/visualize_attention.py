# viz_deformable_queries.py
from rfdetr import RFDETRNano
import torch
import torch.nn as nn
import torchvision.transforms.functional as Fv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from contextlib import contextmanager

# ------------------------
# 0) Preprocessing = your pipeline
# ------------------------


def prepare_image_like_model(img_path, model, device="cpu"):
    """Match your model's preprocessing exactly."""
    img = Image.open(img_path).convert("RGB")

    # to_tensor -> values in [0,1], CHW
    img_t = Fv.to_tensor(img)

    # the same sanity checks as your pipeline
    if (img_t > 1).any():
        raise ValueError(
            "Image has pixel values above 1. Ensure values are scaled to [0,1]."
        )
    if img_t.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {img_t.shape[0]}.")

    # normalize first, then resize to (res, res), mirroring your code
    img_t = img_t.to(device)
    img_t = Fv.normalize(img_t, mean=model.means, std=model.stds)

    # square resize (distorts AR, but matches your model)
    res = int(model.model.resolution)
    img_t = Fv.resize(img_t, [res, res])  # CHW tensor resize

    return img_t.unsqueeze(0), img  # (1,3,res,res), and original PIL for visualization


# ------------------------
# 1) Find MSDeformAttn in decoder
# ------------------------
def find_decoder_msda(model: nn.Module):
    cands = []
    for name, m in model.named_modules():
        if m.__class__.__name__.lower() == "msdeformattn" and (
            "decoder" in name or "dec" in name
        ):
            cands.append((name, m))
    if not cands:
        raise RuntimeError(
            "Decoder MSDeformAttn not found. Inspect model to point to the cross-attn module explicitly."
        )
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]  # deepest one


# ------------------------
# 2) Spatial HxW hook (pre-flatten conv/feat)
# ------------------------
def register_spatial_hook(model: nn.Module, stash: dict):
    def hook_fn(module, inp, out):
        # out: (B, C, H, W)
        _, _, H, W = out.shape
        stash["spatial_hw"] = (H, W)

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and "input_proj" in name:
            return m.register_forward_hook(hook_fn)
    return None  # fallback: infer square via HW later


# ------------------------
# 3) Capture MSDeformAttn internals
# ------------------------
@contextmanager
def capture_msda(msda_module, stash: dict):
    """
    Capture either:
      - sampling_locations & attention_weights provided as MSDeformAttn inputs (Deformable DETR API), OR
      - internal linear outputs sampling_offsets & attention_weights, plus reference_points & spatial_shapes,
        then reconstruct sampling_locations.
    Stashes:
      stash["attn"] -> (B, Q, Hh, L, P) attention weights (after softmax)
      stash["loc"]  -> (B, Q, Hh, L, P, 2) sampling locations in normalized [0,1] coords per level
      stash["spatial_shapes"] -> (L, 2) tensor of (H_l, W_l) if available
    """
    # Try to hook internal linear heads
    attn_linear = getattr(msda_module, "attention_weights", None)
    off_linear = getattr(msda_module, "sampling_offsets", None)

    handles = []

    # Forward hook at module level to sniff inputs (reference_points, spatial_shapes, etc.)
    def msda_forward_hook(module, inputs, output):
        # Heuristics: look for tensors among inputs with shapes matching known args
        # spatial_shapes: (L, 2)
        spatial_shapes = None
        reference_points = None
        sampling_locations = None
        attention_weights_in = None

        for x in inputs:
            if isinstance(x, torch.Tensor):
                if x.dim() == 2 and x.shape[1] == 2:  # (L, 2)
                    spatial_shapes = x
                elif x.dim() == 4 and x.shape[-1] in (2, 4):  # (B, Q, L, 2 or 4)
                    reference_points = x
                elif x.dim() == 6 and x.shape[-1] == 2:  # (B, Q, Hh, L, P, 2)
                    sampling_locations = x
                elif (
                    x.dim() in (4, 5)
                    and x.shape[-1] not in (2, 4)
                    and x.dtype.is_floating_point
                ):
                    # Could be attention_weights (B,Q,Hh,L,P) or a flattened variant
                    attention_weights_in = x

        if sampling_locations is not None and attention_weights_in is not None:
            # Best case: API passes both in
            stash["loc"] = sampling_locations.detach()
            # Normalize weights over last dim if not already
            attn = attention_weights_in
            if attn.dim() == 5:  # (B,Q,Hh,L,P)
                # Softmax typically applied over Hh*L*P; but often it's already softmaxed.
                # We renormalize over P for visualization convenience.
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
                stash["attn"] = attn.detach()
            else:
                stash["attn"] = None
        else:
            # Will try to reconstruct from internal linear heads below
            stash["reference_points"] = (
                reference_points.detach() if reference_points is not None else None
            )
            stash["spatial_shapes"] = (
                spatial_shapes.detach() if spatial_shapes is not None else None
            )

    handles.append(msda_module.register_forward_hook(msda_forward_hook))

    # Hooks on internal linear heads to grab their outputs
    def attn_head_hook(module, inp, out):
        stash["_raw_attn_linear"] = out.detach()  # (B, Q, Hh*L*P)

    def off_head_hook(module, inp, out):
        stash["_raw_off_linear"] = out.detach()  # (B, Q, Hh*L*P*2)

    if attn_linear is not None:
        handles.append(attn_linear.register_forward_hook(attn_head_hook))
    if off_linear is not None:
        handles.append(off_linear.register_forward_hook(off_head_hook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def reconstruct_sampling(msda, stash):
    """
    Build sampling locations & attention weights from MSDeformAttn internals.
    Returns:
      attn: (B, Q, Hh, L, P)
      loc : (B, Q, Hh, L, P, 2)   in normalized [0,1]
      spatial_shapes: (L, 2) or None
    """
    # Fast path: already captured ready-to-use tensors
    if "attn" in stash and stash.get("loc", None) is not None:
        return stash["attn"], stash["loc"], stash.get("spatial_shapes")

    raw_attn = stash.get("_raw_attn_linear")  # (B, Q, Hh*L*P)
    raw_off = stash.get("_raw_off_linear")  # (B, Q, Hh*L*P*2)
    ref_pts = stash.get("reference_points")  # (B, Q, L, 2) or (B, Q, L, 4)
    spatial_shapes = stash.get("spatial_shapes")  # (L, 2) as (H_l, W_l)

    if raw_attn is None or raw_off is None or ref_pts is None:
        raise RuntimeError(
            "Missing tensors to reconstruct deformable sampling "
            "(need attention/offests/reference_points)."
        )

    # Extract or infer Hh/L/P
    n_heads = getattr(msda, "n_heads", None)
    n_levels = getattr(msda, "n_levels", None)
    n_points = getattr(msda, "n_points", None)
    B, Q, totalA = raw_attn.shape
    if n_heads is None or n_levels is None or n_points is None:
        # Infer from totalA = Hh*L*P
        L_true = int(spatial_shapes.shape[0]) if (spatial_shapes is not None) else None
        guess = None
        for Hh in (4, 8, 16):
            for L in (1, 2, 3, 4):
                if L_true is not None and L != L_true:
                    continue
                if totalA % (Hh * L) == 0:
                    P = totalA // (Hh * L)
                    guess = (Hh, L, P)
                    break
            if guess:
                break
        if guess is None:
            raise RuntimeError(f"Cannot factorize Hh/L/P from total={totalA}.")
        n_heads, n_levels, n_points = guess

    # Reshape and softmax weights
    attn = raw_attn.view(B, Q, n_heads, n_levels, n_points).softmax(
        dim=4
    )  # (B,Q,Hh,L,P)
    offsets = raw_off.view(B, Q, n_heads, n_levels, n_points, 2)  # (B,Q,Hh,L,P,2)

    # Handle 2-D vs 4-D reference points
    if ref_pts.shape[-1] == 2:
        if spatial_shapes is None:
            raise RuntimeError("spatial_shapes required when reference_points are 2-D.")
        # spatial_shapes: (L, 2) = (H_l, W_l)
        H_l = spatial_shapes[:, 0].float()
        W_l = spatial_shapes[:, 1].float()
        # Build (1,1,1,L,1,2) normalizer as (W_l, H_l) to match (x,y)
        norm = torch.stack([W_l, H_l], dim=1)[None, None, None, :, None, :].to(
            offsets.device
        )
        ref_xy = ref_pts[:, :, None, :, None, :]  # (B,Q,1,L,1,2)
        loc = ref_xy + offsets / (norm + 1e-6)  # (B,Q,Hh,L,P,2)
    elif ref_pts.shape[-1] == 4:
        # Deformable DETR box-referenced formula:
        # loc = ref_xy + (offsets / n_points) * 0.5 * ref_wh
        ref_xy = ref_pts[..., :2]  # (B,Q,L,2)
        ref_wh = ref_pts[..., 2:]  # (B,Q,L,2)
        ref_xy = ref_xy[:, :, None, :, None, :]  # (B,Q,1,L,1,2)
        ref_wh = ref_wh[:, :, None, :, None, :]  # (B,Q,1,L,1,2)
        loc = ref_xy + offsets * (0.5 / float(n_points)) * ref_wh
    else:
        raise RuntimeError(
            f"Unsupported reference_points last dim: {ref_pts.shape[-1]} (expected 2 or 4)"
        )

    return (
        attn.detach(),
        loc.detach(),
        spatial_shapes.detach() if spatial_shapes is not None else None,
    )


# ------------------------
# 4) Drive a forward pass & capture
# ------------------------
@torch.no_grad()
def forward_and_capture_msda(model, inp):
    msda = find_decoder_msda(model)
    stash = {}
    size_handle = register_spatial_hook(model, stash)

    model.eval()
    with capture_msda(msda, stash):
        outputs = model(inp)

    if size_handle is not None:
        size_handle.remove()

    attn, loc, spatial_shapes = reconstruct_sampling(msda, stash)
    return outputs, attn, loc, spatial_shapes, stash.get("spatial_hw")


# ------------------------
# 5) Utilities
# ------------------------
def get_logits(outputs):
    if isinstance(outputs, dict) and "pred_logits" in outputs:
        return outputs["pred_logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    raise RuntimeError("Could not find class logits in outputs.")


def objectness_from_logits(logits):
    probs = logits.softmax(-1)
    return 1.0 - probs[..., -1]  # 1 - p(no_object)


def overlay_points(ax, img_pil, points_xy, weights=None, title=None):
    ax.imshow(img_pil)
    if points_xy.size == 0:
        ax.set_axis_off()
        return
    if weights is None:
        weights = np.ones((points_xy.shape[0],), dtype=np.float32)
    # scale points by weights for visibility
    sizes = 100.0 * (weights / (weights.max() + 1e-6))
    ax.scatter(
        points_xy[:, 0],
        points_xy[:, 1],
        s=sizes,
        marker="o",
        linewidths=0.5,
        edgecolors="white",
    )
    if title:
        ax.set_title(title)
    ax.axis("off")


def norm_to_image_xy(points_norm, img_w, img_h):
    # points_norm: (..., 2) in [0,1] (x,y)
    pts = points_norm.clone()
    pts[..., 0] = pts[..., 0] * img_w
    pts[..., 1] = pts[..., 1] * img_h
    return pts


# ------------------------
# 6) Public API: visualize top-K queries (deformable)
# ------------------------
def visualize_queries(model, image_path, topk=6, per_head=False):
    """
    Shows the deformable sampling points each top-K query uses (weighted by attention).
    If per_head=True, shows one panel per head for each query; otherwise, averages heads.
    """
    # 1) preprocess
    inp, img_pil = prepare_image_like_model(image_path, model)
    model = model.model.model
    img_w, img_h = img_pil.size

    # 2) forward & capture deformable internals
    outputs, attn, loc, spatial_shapes, _ = forward_and_capture_msda(
        model, inp
    )  # attn: (B,Q,Hh,L,P), loc: (B,Q,Hh,L,P,2)

    # 3) pick top-K queries by objectness
    logits = get_logits(outputs)
    objness = objectness_from_logits(logits)  # (B,Q)
    B, Q = objness.shape
    k = min(topk, Q)
    scores, q_idx = torch.topk(objness[0], k)

    # 4) plotting
    n_heads = attn.shape[2]
    n_panels = (k * n_heads) if per_head else k
    cols = min(n_panels, 6)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    panel = 0
    for qid, sc in zip(q_idx.cpu().tolist(), scores.cpu().tolist()):
        # gather this query's points
        w_q = attn[0, qid]  # (Hh,L,P)
        loc_q = loc[0, qid]  # (Hh,L,P,2)

        if per_head:
            for h in range(n_heads):
                w_h = w_q[h]  # (L,P)
                loc_h = loc_q[h]
                # average over levels to a single set of points (concat levels)
                w_hp = w_h.reshape(-1).cpu().numpy()
                loc_hp = loc_h.reshape(-1, 2)
                pts_img = norm_to_image_xy(loc_hp, img_w, img_h).cpu().numpy()
                overlay_points(
                    axes[panel],
                    img_pil,
                    pts_img,
                    w_hp,
                    title=f"Q{qid} H{h} | obj={sc:.2f}",
                )
                panel += 1
        else:
            # average weights over heads, then concat levels
            w_mean = w_q.mean(0)  # (L,P)
            loc_mean = loc_q.mean(0)  # (L,P,2)  (averaging locs is okay for viz)
            w_lp = w_mean.reshape(-1).cpu().numpy()
            loc_lp = loc_mean.reshape(-1, 2)
            pts_img = norm_to_image_xy(loc_lp, img_w, img_h).cpu().numpy()
            overlay_points(
                axes[panel], img_pil, pts_img, w_lp, title=f"Q{qid} | obj={sc:.2f}"
            )
            panel += 1

    # tidy
    for i in range(panel, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    # Save figure
    fig.savefig("visualize_queries.png", bbox_inches="tight")
    return fig


model = RFDETRNano(num_classes=4, device="cpu")

checkpoint = torch.load("output/checkpoint.pth", map_location="cpu", weights_only=False)
model.model.model.load_state_dict(checkpoint["model"], strict=True)

image_path = "../example.jpg"
fig = visualize_queries(model, image_path)

fig = visualize_queries(model, image_path, per_head=True)
