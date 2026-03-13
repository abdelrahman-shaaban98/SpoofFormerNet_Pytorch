from typing import  Tuple
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from depth_estimator import estimate_depth
from model import build_spoof_former_net


COLOR_MEAN = (0.5, 0.5, 0.5)
COLOR_STD  = (0.5, 0.5, 0.5)
DEPTH_MEAN = (0.5,)
DEPTH_STD  = (0.5,)


def preprocess_image(path: str, image_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    rgb_img = Image.open(path).convert("RGB")
    rgb_img = TF.resize(rgb_img, (image_size, image_size))
    rgb_t   = TF.to_tensor(rgb_img)
    rgb_t   = TF.normalize(rgb_t, mean=COLOR_MEAN, std=COLOR_STD)
    rgb_t = rgb_t.unsqueeze(0)          # 1, 3, H, W

    depth_np = estimate_depth(path)
    # depth_np = TF.resize(depth_np, (image_size, image_size))
    depth_t   = TF.to_tensor(depth_np).mean(dim=0, keepdim=True)   # 1, H, W
    depth_t = TF.resize(depth_t, (image_size, image_size))
    depth_t   = TF.normalize(depth_t, mean=DEPTH_MEAN, std=DEPTH_STD)
    depth_t = depth_t.unsqueeze(0)          # 1, 1, H, W

    return rgb_t, depth_t



def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg_variant = ckpt.get("cfg", {}).get("model_variant")
    model = build_spoof_former_net(cfg_variant).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Variant    : {cfg_variant}")
    print(f"  Saved at   : epoch {epoch}")

    if "val_metrics" in ckpt:
        m = ckpt["val_metrics"]
        print(f"  Val ACER   : {m.get('acer', '?'):.4f}   "
              f"AUC: {m.get('auc', '?'):.4f}")
        
    return model


def save_model(save_path, epoch, model, optimizer, cfg, val_metrics):
    torch.save(
        {
            "epoch"       : epoch,
            "model_state" : model.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "cfg"         : cfg,
            "val_metrics" : val_metrics,
        },
        save_path,
        # save_dir / f"checkpoint_epoch_{epoch:04d}.pt",
    )


def print_training_progress(epoch, train_metrics, val_metrics, eta, flag):
    print(
        f"{epoch:>4}  "
        f"{train_metrics['loss']:>8.4f}  {train_metrics['accuracy']:>7.4f}  "
        f"{val_metrics['loss']:>9.4f}  {val_metrics['accuracy']:>8.4f}  "
        f"{val_metrics['apcer']:>6.4f}  {val_metrics['bpcer']:>6.4f}  "
        f"{val_metrics['acer']:>6.4f}  {val_metrics['auc']:>7.4f}  "
        f"{eta:>6}{flag}"
    )


def count_params(model: nn.Module) -> str:
    # n = sum(p.numel() for p in model.parameters())
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return f"{n / 1e6:.2f} M"
