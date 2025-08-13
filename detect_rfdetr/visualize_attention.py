from rfdetr import RFDETRNano
import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.functional as Fv

############################################
# 1) Utilities
############################################


def overlay_heatmap_on_image(img_pil, heatmap_0to1, alpha=0.45):
    """img_pil: PIL RGB, heatmap_0to1: HxW in [0,1]"""
    hm = (heatmap_0to1 * 255).astype(np.uint8)
    hm = Image.fromarray(hm).convert("L").resize(img_pil.size, Image.BILINEAR)
    hm_color = Image.fromarray(
        plt.cm.jet(np.array(hm))[:, :, :3][:, :, ::-1]
    )  # BGR→RGB fix
    # blend
    return Image.blend(img_pil.convert("RGB"), hm_color.convert("RGB"), alpha)


def find_last_cross_attn_in_decoder(model):
    """
    Returns (module, layer_name) for the *last* nn.MultiheadAttention that is likely
    the decoder's cross-attention. Works for many DETR-style codebases.
    """
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, nn.MultiheadAttention) and (
            "decoder" in name or "dec" in name
        ):
            candidates.append((name, m))
    if not candidates:
        raise RuntimeError(
            "Could not find a decoder MultiheadAttention. "
            "Search your model for the cross-attn module and point the hook to it."
        )
    # pick the last one (deepest)
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1], candidates[-1][0]


############################################
# 2) Hook setup
############################################


class AttnCatcher:
    def __init__(self):
        self.cross_attn_weights = None  # (B, heads, Q, HW)
        self.spatial_hw = None  # (H, W)
        self.handles = []

    def clear(self):
        self.cross_attn_weights = None
        self.spatial_hw = None


def register_hooks(model, catcher: AttnCatcher):
    # a) cross-attention hook (forward hook that sees (attn_output, attn_weights))
    cross_attn_module, name = find_last_cross_attn_in_decoder(model)

    def cross_attn_hook(module, inp, out):
        # torch.nn.MultiheadAttention returns (attn_output, attn_weights)
        attn_weights = out[
            1
        ]  # (Q, B, heads, HW) or (B, heads, Q, HW) depending on impl
        # normalize shape to (B, heads, Q, HW)
        if attn_weights.dim() == 4:
            # guess ordering
            if (
                attn_weights.shape[0] != attn_weights.shape[1]
            ):  # often (B, heads, Q, HW)
                catcher.cross_attn_weights = attn_weights.detach().cpu()
            else:
                # fallback: (Q, B, heads, HW) → (B, heads, Q, HW)
                attn_weights = attn_weights.permute(1, 2, 0, 3).contiguous()
                catcher.cross_attn_weights = attn_weights.detach().cpu()
        else:
            raise RuntimeError("Unexpected attention weight shape.")

    catcher.handles.append(cross_attn_module.register_forward_hook(cross_attn_hook))

    # b) spatial size hook — most DETR variants apply a 1x1 conv `input_proj` to backbone feat (B,C,H,W)
    input_proj = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and "input_proj" in name:
            input_proj = m
    if input_proj is None:
        print("WARNING: input_proj not found. You will need to set H,W manually.")
    else:

        def input_proj_hook(module, inp, out):
            # out: (B, C, H, W)
            _, _, H, W = out.shape
            catcher.spatial_hw = (H, W)

        catcher.handles.append(input_proj.register_forward_hook(input_proj_hook))


def remove_hooks(catcher: AttnCatcher):
    for h in catcher.handles:
        h.remove()
    catcher.handles = []


def _prepare_image_like_model(img_path, model, device):
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


@torch.no_grad()
def visualize_queries(model, image_path, topk=6, device="cpu"):
    inner_model = model.model.model
    inner_model.eval()

    # prepare input exactly as the model expects
    inp, img_pil = _prepare_image_like_model(
        image_path, model, device
    )  # (1,3,res,res), PIL

    # hook cross-attn and HxW
    catcher = AttnCatcher()
    register_hooks(inner_model, catcher)

    # forward
    outputs = inner_model(inp)

    # class logits (adjust if your dict keys differ)
    if isinstance(outputs, dict) and "pred_logits" in outputs:
        logits = outputs["pred_logits"]  # (B,Q,C+1)
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        remove_hooks(catcher)
        raise RuntimeError("Couldn't find class logits; adapt to your model's output.")

    # objectness = 1 - p(no_object)
    probs = logits.softmax(-1)
    objness = 1.0 - probs[..., -1]
    B, Q = objness.shape
    k = min(topk, Q)
    scores, q_idx = torch.topk(objness[0], k)

    # attention weights captured from last decoder cross-attn
    attn = catcher.cross_attn_weights  # (B, heads, Q, HW)
    if attn is None:
        remove_hooks(catcher)
        raise RuntimeError(
            "Cross-attention not captured. Ensure hook targets decoder cross-attn."
        )

    # infer spatial grid (should match your post-resize feature map)
    if catcher.spatial_hw is not None:
        H, W = catcher.spatial_hw
    else:
        HW = attn.shape[-1]
        s = int(np.sqrt(HW))
        if s * s != HW:
            remove_hooks(catcher)
            raise RuntimeError("Cannot infer HxW; hook a (B,C,H,W) tensor pre-flatten.")
        H, W = s, s

    # average over heads → (Q, H, W)
    # attn: (B, heads, Q, HW)
    attn_q_hw = attn[0].mean(0)  # (heads, Q, HW)
    attn_q_hw = attn_q_hw.mean(0)  # (Q, HW)
    attn_maps = attn_q_hw.view(Q, H, W).cpu().numpy()

    # plot top-k query heatmaps
    n = k
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, qid, sc in zip(axes, q_idx.cpu().tolist(), scores.cpu().tolist()):
        hm = attn_maps[qid]
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
        blended = overlay_heatmap_on_image(img_pil, hm, alpha=0.45)
        ax.imshow(blended)
        ax.set_title(f"Query {qid} | obj={sc:.2f}")
        ax.axis("off")
    plt.tight_layout()
    remove_hooks(catcher)
    plt.show()


############################################
# 4) Usage
############################################
# Example:
#   - If you have a FacebookResearch DETR-style model in `model`
#   - Or a LW-DETR variant with similar module names
#
# model = load_your_trained_model()  # you supply this
# visualize_queries(model, "your_image.jpg", device="cuda", topk=6, img_size=800)


model = RFDETRNano(num_classes=4)


checkpoint = torch.load("output/checkpoint.pth", map_location="cpu", weights_only=False)
inner_model = model.model.model
inner_model.load_state_dict(checkpoint["model"], strict=True)
inner_model.eval()

image_path = "../example.jpg"
visualize_queries(model, image_path)
