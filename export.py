import sys
import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

import torch
import torch.nn as nn

from utils import load_model, count_params


IMAGE_SIZE = 256


def _verify_torchscript(
    model:       nn.Module,
    output_path:  str,
    rgb:         torch.Tensor,
    depth:       torch.Tensor,
    device:      torch.device,
    atol:        float = 1e-4,
) -> bool:
    """
    Compare eager-mode and TorchScript outputs on the same dummy inputs.
    Returns True if outputs match within atol.
    """
    model.eval()
    torchscript_model = torch.jit.load(output_path, map_location=device)
    torchscript_model.eval()

    with torch.no_grad():
        eager_out  = torch.softmax(
            model(rgb.to(device), depth.to(device)), dim=-1
        ).cpu()
        script_out = torch.softmax(
            torchscript_model(rgb.to(device), depth.to(device)), dim=-1
        ).cpu()

    diff = float((eager_out - script_out).abs().max())
    ok       = diff <= atol
    status   = "OK ✓" if ok else f"MISMATCH"
    print(f"  Verification : {status}  (diff = {diff:.2e},  atol = {atol})")

    return ok


def _verify_onnx(
    model:      nn.Module,
    onnx_path:  str,
    rgb:        torch.Tensor,
    depth:      torch.Tensor,
    device:     torch.device,
    atol:       float = 1e-4,
):
    """Compare PyTorch and ONNX Runtime outputs on the same dummy inputs."""

    # PyTorch reference
    with torch.no_grad():
        pt_out = torch.softmax(model(rgb.to(device), depth.to(device)), dim=-1)
    
    pt_np = pt_out.cpu().numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(onnx_path,
                                providers=["CUDAExecutionProvider"])
    ort_inputs = {
        sess.get_inputs()[0].name: rgb.cpu().numpy().astype("float32"),
        sess.get_inputs()[1].name: depth.cpu().numpy().astype("float32"),
    }
    ort_out = sess.run(None, ort_inputs)[0]

    max_diff = float(np.abs(pt_np - ort_out).max())
    status   = "OK ✓" if max_diff <= atol else f"MISMATCH  (diff {max_diff:.2e})"
    print(f"  Output verification  : {status}")

    if max_diff > atol:
        print("  [WARNING] Outputs differ")
        print(pt_np)
        print(ort_out)


def export_torchscript():
    pass



def export_onnx():
    pass




def export(
    checkpoint_path: str,
    export_to: str,
    output_path:     str   = "checkpoints/spoof_former_net",
    device:      str   = "cuda",
    verify:          bool  = True,
):
    device = torch.device(device)

    print("\n" + "=" * 65)
    print("  SpoofFormerNet — TorchScript Export")
    print("=" * 65)

    # ── Load weights ──────────────────────────────────────────────────
    model = load_model(checkpoint_path, device)
    print(f"  Parameters   : {count_params(model)}")

    # Dummy inputs
    rgb_dummy   = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    depth_dummy = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)


    if export_to == "torchscript":
        print(f"\n  Method       : torch.jit.trace") 

        output_path = output_path + ".torchscript.pt"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            script_mod = torch.jit.trace(
                model,
                (rgb_dummy, depth_dummy),
                strict=True,          # raise on tensor-shape surprises
                check_trace=True,     # re-run and compare to catch non-determinism
                check_tolerance=1e-4,
            )


        # torch.jit.freeze folds constants (BatchNorm statistics, bias) into
        # the graph, removing the need to carry them as separate parameters.
        script_mod = torch.jit.freeze(script_mod)
        # optimize_for_inference fuses Conv+BN, eliminates dead nodes, etc.
        script_mod = torch.jit.optimize_for_inference(script_mod)

        torch.jit.save(script_mod, output_path)
    
    elif export_to == "onnx":
        print(f"\n  Method       : onnx") 

        output_path = output_path + ".onnx"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            (rgb_dummy, depth_dummy),
            output_path,
            export_params   = True,
            do_constant_folding = True,
            input_names     = ["rgb", "depth"],
            output_names    = ["logits"],
        )

        # Validate ONNX graph structure 
        try:
            onnx.checker.check_model(output_path)
            print("  ONNX check   : passed ✓")
        except Exception as exc:
            print(f"  ONNX check   : FAILED — {exc}")
            sys.exit(1)


    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Saved        : {output_path}  ({size_mb:.1f} MB)")

    if verify:
        if export_to == "torchscript":
            print("\n  Loading saved file for verification ...")
            _verify_torchscript(model, output_path, rgb_dummy, depth_dummy, device)
        elif export_to == "onnx":
            _verify_onnx(model, output_path, rgb_dummy, depth_dummy, device)


    print("\n" + "=" * 65)
    print(f"  TorchScript model → {output_path}")
    print("=" * 65 + "\n")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Export SpoofFormerNet checkpoint to TorchScript"
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path to the .pt checkpoint")
    p.add_argument("--export-to", required=True,
                   help="Export to (torchscript|onnx)")
    p.add_argument("--output",        default="checkpoints/spoof_former_net",
                   help="Destination file (default: checkpoints/spoof_former_net)")    
    p.add_argument("--device", default="cuda",
                   help="'cpu' or 'cuda' (default: cuda)")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip numerical verification after saving")
    
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    export(
        checkpoint_path = args.checkpoint,
        export_to       = args.export_to,
        output_path     = args.output,
        device          = args.device,
        verify          = not args.no_verify,
    )