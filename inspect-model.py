import torch


def tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()


def get_total_size(obj):
    if isinstance(obj, dict):
        return sum(get_total_size(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(get_total_size(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return tensor_size(obj)
    elif hasattr(obj, "state_dict"):
        return get_total_size(obj.state_dict())
    elif isinstance(obj, torch.nn.Module):
        return get_total_size(obj.state_dict())
    return 0


for path in [
    "runs/detect/bird_head_yolov8n/weights/best.pt",
    "runs/detect/bird_head_yolov8n_debug/weights/best.pt",
]:
    print(path)
    ckpt = torch.load(path, weights_only=False)
    for k, v in ckpt.items():
        size_bytes = get_total_size(v)
        print(f"{k:<15} {size_bytes / 1e6:.2f} MB")
