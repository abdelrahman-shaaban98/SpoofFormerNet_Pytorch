from typing import Tuple
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import torch


COLOR_MEAN = (0.5, 0.5, 0.5)
COLOR_STD  = (0.5, 0.5, 0.5)
DEPTH_MEAN = (0.5,)
DEPTH_STD  = (0.5,)

LABEL_TEXT  = {0: "FAKE", 1: "REAL"}
LABEL_COLOR = {0: "#FF4B4B", 1: "#4BFF91"}   # red for fake, green for real


def denormalize(tensor: torch.Tensor, mean: Tuple, std: Tuple) -> np.ndarray:
    """Undo normalisation and return a HxWxC uint8 array."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    img = tensor.cpu().clone() * std + mean     
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()  # C,H,W → H,W,C
    # print(img.shape)
    return (img * 255).astype(np.uint8)



def visualize_batch(
    rgb:    torch.Tensor,         # B, 3, H, W
    depth:  torch.Tensor,         # B, 1, H, W
    labels: torch.Tensor,         # B
    save_path: str = "images/batch_visualization.png",
):
    B = rgb.shape[0]

    # Figure layout
    fig = plt.figure(figsize=(B * 4, 8.5), facecolor="#0D0D0D")

    fig.suptitle(
        "SpoofFormerNet — Training Batch Preview",
        fontsize=16, fontweight="bold", color="#E8E8E8",
        y=0.95, fontfamily="monospace",
    )


    # Two rows, B columns; extra bottom margin for label badges
    gs = GridSpec(
        2, B,
        figure=fig,
        hspace=0.08,
        wspace=0.06,
        left=0.02, right=0.98,
        top=0.9,  bottom=0.1,
    )
    
    for col in range(B):
        label_id = int(labels[col].item())
        label_text      = LABEL_TEXT[label_id]
        label_color  = LABEL_COLOR[label_id]

        # Row 0: RGB 
        ax_rgb = fig.add_subplot(gs[0, col])
        rgb_np = denormalize(rgb[col], COLOR_MEAN, COLOR_STD)
        # rgb_np = rgb[col]

        ax_rgb.imshow(rgb_np)

        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        for spine in ax_rgb.spines.values():
            spine.set_edgecolor(label_color)
            spine.set_linewidth(2.5)

        if col == 0:
            ax_rgb.set_ylabel(
                "RGB", color="#E8E8E8",
            )

        # Row 1: Depth 
        ax_dep = fig.add_subplot(gs[1, col])
        dep_np = denormalize(depth[col], DEPTH_MEAN, DEPTH_STD)   # H, W, 1

        dep_np = dep_np[:, :, 0]                                 
        ax_dep.imshow(dep_np, cmap="plasma")
        
        ax_dep.set_xticks([]); ax_dep.set_yticks([])
        for spine in ax_dep.spines.values():
            spine.set_edgecolor(label_color)
            spine.set_linewidth(2.5)

        if col == 0:
            ax_dep.set_ylabel(
                "Depth", color="#E8E8E8",
            )

        # Label badge below the depth image 
        ax_dep_pos = ax_dep.get_position()
        text_x = ax_dep_pos.x0 + ax_dep_pos.width / 2
        text_y = ax_dep_pos.y0 - 0.055

        fig.text(
            text_x, text_y, f"● {label_text}",
            ha="center", va="top",
            fontsize=12, fontweight="bold", fontfamily="monospace",
            color=label_color,
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualize_batch] Saved → {save_path}")


def visualise_result(
    rgb_tensor:   torch.Tensor,
    depth_tensor: torch.Tensor,
    label_id:     int,
    confidence:   float,
    probs:        torch.Tensor,
    image_path:   str,
    save_path:    str = "images/inference_result.png",
):

    rgb_np   = denormalize(rgb_tensor,   COLOR_MEAN, COLOR_STD)
    depth_np = denormalize(depth_tensor, DEPTH_MEAN, DEPTH_STD)

    prediction_color = LABEL_COLOR[label_id]
    prediction_text  = LABEL_TEXT[label_id]

    fig = plt.figure(figsize=(11, 5), facecolor="#0D0D0D")
    gs  = GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1, 1, 1.15],
        wspace=0.08,
        left=0.03, right=0.97,
        top=0.82,  bottom=0.08,
    )

    # RGB panel 
    ax_rgb = fig.add_subplot(gs[0])
    ax_rgb.imshow(rgb_np)
    ax_rgb.set_title("RGB Input", color="#AAAAAA",
                     fontfamily="monospace", fontsize=10, pad=6)
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
    
    for sp in ax_rgb.spines.values():
        sp.set_edgecolor(prediction_color); sp.set_linewidth(2.5)

    # Depth panel 
    ax_dep = fig.add_subplot(gs[1])
    ax_dep.imshow(depth_np, cmap="plasma")
    ax_dep.set_title("Depth Map", color="#AAAAAA",
                     fontfamily="monospace", fontsize=10, pad=6)
    ax_dep.set_xticks([]); ax_dep.set_yticks([])
    
    for sp in ax_dep.spines.values():
        sp.set_edgecolor(prediction_color); sp.set_linewidth(2.5)

    #  Confidence bar chart 
    ax_bar = fig.add_subplot(gs[2])
    ax_bar.set_facecolor("#1A1A1A")
    
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#333333")

    classes = ["FAKE", "REAL"]
    colors  = [LABEL_COLOR[0], LABEL_COLOR[1]]
    vals    = [probs[0].item() * 100, probs[1].item() * 100]

    bars = ax_bar.barh(classes, vals, color=colors, height=0.45, zorder=3)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Confidence (%)", color="#AAAAAA",
                      fontfamily="monospace", fontsize=9)
    ax_bar.tick_params(colors="#AAAAAA", labelsize=10)
    ax_bar.xaxis.label.set_color("#AAAAAA")
    ax_bar.grid(axis="x", color="#333333", linestyle="--", linewidth=0.6, zorder=0)

    for bar, val in zip(bars, vals):
        ax_bar.text(
            min(val + 1.5, 95), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", color="white",
            fontfamily="monospace", fontsize=10, fontweight="bold",
        )

    ax_bar.set_title("Class Probabilities", color="#AAAAAA",
                     fontfamily="monospace", fontsize=10, pad=6)
    for tick in ax_bar.get_yticklabels():
        tick.set_fontfamily("monospace")
        tick.set_fontsize(11)
        tick.set_color(colors[classes.index(tick.get_text())])

    # Main title / prediction 
    fname = Path(image_path).name
    fig.suptitle(
        f"SpoofFormerNet    {fname}",
        color="#CCCCCC", fontfamily="monospace", fontsize=12,
        fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.90,
        f"Prediction:  {prediction_text}  ({confidence * 100:.1f}% confidence)",
        ha="center", va="top",
        fontfamily="monospace", fontsize=15, fontweight="bold",
        color=prediction_color,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved visualisation → {save_path}")

