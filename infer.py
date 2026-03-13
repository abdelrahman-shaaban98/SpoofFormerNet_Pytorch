import argparse

import torch
import torch.nn.functional as F

from utils import preprocess_image, load_model
from visualize import visualise_result


LABEL_TEXT  = {0: "FAKE", 1: "REAL"}
IMAGE_SIZE = 256
THRESHOLD = 0.5

def infer(
    image_path:      str,
    infer_type:      str,
    model_path:      str,
    device_str:      str = "auto",
    save_vis:        bool = True,
    vis_path:        str = "images/inference_result.png",
) -> None:
    """
    Run inference on a single image.
    """
    print("=" * 65)
    print("  SpoofFormerNet — Inference")
    print("=" * 65)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    rgb_tensor, depth_tensor = preprocess_image(image_path, IMAGE_SIZE)
    rgb_tensor, depth_tensor = rgb_tensor.to(device), depth_tensor.to(device)

    with torch.no_grad():
        if infer_type == "torch":
            model = load_model(model_path, device)
            logits = model(rgb_tensor, depth_tensor)
        
        elif infer_type == "torchscript":
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
            logits = model(rgb_tensor, depth_tensor)
        
        elif infer_type == "onnx":
            pass #TODO to implement
        
        else:
            pass #TODO Through an exception

        probs  = F.softmax(logits, dim=-1).squeeze(0) 


    prob_real  = probs[1].item()
    prob_fake  = probs[0].item()
    label_id   = int(prob_real >= THRESHOLD)
    confidence = prob_real if label_id == 1 else prob_fake
    verdict    = LABEL_TEXT[label_id]


    print("\n" + "-" * 65)
    print(f"  {'Verdict':<18}: {verdict}")
    print(f"  {'Confidence':<18}: {confidence * 100:.2f}%")
    print(f"  {'P(REAL)':<18}: {prob_real * 100:.2f}%")
    print(f"  {'P(FAKE)':<18}: {prob_fake * 100:.2f}%")
    print(f"  {'Threshold':<18}: {THRESHOLD}")
    print("-" * 65 + "\n")


    if save_vis:
        vis_path = vis_path.replace(".png", f"_{infer_type}.png")
        visualise_result(
            rgb_tensor   = rgb_tensor.squeeze(0).cpu(),
            depth_tensor = depth_tensor.squeeze(0).cpu(),
            label_id     = label_id,
            confidence   = confidence,
            probs        = probs.cpu(),
            image_path   = image_path,
            save_path    = vis_path,
        )


def _parse_args():
    p = argparse.ArgumentParser(
        description="SpoofFormerNet — single-image inference"
    )

    p.add_argument("--image",      required=True,
                   help="Path to the input face image (.jpg|.png …)")
    p.add_argument("--infer-type", required=True,
                   help="Path to the input face image (torch|torchscript|onnx)")
    p.add_argument("--model-path", required=True,
                   help="Path to the .pt|.torchscript.pt|.onnx checkpoint file")
    p.add_argument("--device",     default="auto",
                   help="'auto', 'cpu', or 'cuda' (default: auto).")
    p.add_argument("--no-vis",     action="store_true",
                   help="Skip saving the visualisation image.")
    p.add_argument("--vis-path",   default="images/inference_result.png",
                   help="Where to save the visualisation (default: images/inference_result.png).")
    
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    infer(
        image_path      = args.image,
        infer_type      = args.infer_type,
        model_path      = args.model_path,
        device_str      = args.device,
        save_vis        = not args.no_vis,
        vis_path        = args.vis_path,
    )