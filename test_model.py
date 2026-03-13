import numpy as np

import torch
import onnx
import onnxruntime as ort

from transformer import *
from model import *



def test_ConvFFN():
    model = ConvFFN(dim=64).eval()
    B, H, W, C = 2, 16, 16, 64
    N = H * W # 16 * 16 = 65536

    x = torch.randn(B, N, C)
    out = model(x, H, W)
    print(out.shape) # Expected [2, 16 * 16, 64]

    # To bisualize
    # from torchviz import make_dot
    # dot = make_dot(out, params=dict(model.named_parameters()))
    # dot.render("convffn_graph", format="png")

    # onnx testing
    torch.onnx.export(
        model, 
        (x, H, W), 
        "ConvFNN.onnx", 
        input_names=["x", "H", "W"],
        output_names=["y"],
    )

    model = onnx.load("ConvFNN.onnx")
    onnx.checker.check_model(model)

    sess = ort.InferenceSession("ConvFNN.onnx")
    out = sess.run(
        None,
        {
            "x": x.numpy(),
        },
    )

    print(out[0].shape) # Expected [2, 16 * 16, 64]


def test_WeightedMSA():
    model = WeightedMSA(dim=64).eval()
    B, H, W, C = 2, 16, 16, 64
    N = H * W # 16 * 16 = 256

    x = torch.randn(B, N, C)
    out = model(x)
    print(out.shape) # Expected [2, 16 * 16, 64]

    # onnx testing
    torch.onnx.export(
        model, 
        (x,), 
        "WeightedMSA.onnx", 
        input_names=["x"],
        output_names=["y"],
    )

    model = onnx.load("WeightedMSA.onnx")
    onnx.checker.check_model(model)

    sess = ort.InferenceSession("WeightedMSA.onnx")
    out = sess.run(
        None,
        {
            "x": x.numpy(),
        },
    )

    print(out[0].shape) # Expected [2, 16 * 16, 64]


def test_LocalWindowAttention():

    model = LocalWindowAttention(dim=64).eval()
    B, H, W, C = 2, 16, 16, 64
    N = H * W # 16 * 16 = 256

    x = torch.randn(B, N, C)
    out = model(x, H, W)
    print(out.shape) # Expected [2, 16 * 16, 64]

    # onnx testing
    torch.onnx.export(
        model, 
        (x, H, W), 
        "LocalWindowAttention.onnx", 
        input_names=["x"],
        output_names=["y"],
    )

    model = onnx.load("LocalWindowAttention.onnx")
    onnx.checker.check_model(model)

    sess = ort.InferenceSession("LocalWindowAttention.onnx")
    out = sess.run(
        None,
        {
            "x": x.numpy(),
        },
    )

    print(out[0].shape) # Expected [2, 16 * 16, 64]


def test_SparseGlobalAttention():

    model = SparseGlobalAttention(dim=64).eval()
    B, H, W, C = 2, 16, 16, 64
    N = H * W # 16 * 16 = 256

    x = torch.randn(B, N, C)
    out = model(x)
    print(out.shape) # Expected [2, 16 * 16, 64]

    # onnx testing
    torch.onnx.export(
        model, 
        (x,), 
        "SparseGlobalAttention.onnx", 
        input_names=["x"],
        output_names=["y"],
    )

    model = onnx.load("SparseGlobalAttention.onnx")
    onnx.checker.check_model(model)

    sess = ort.InferenceSession("SparseGlobalAttention.onnx")
    out = sess.run(
        None,
        {
            "x": x.numpy(),
        },
    )

    print(out[0].shape) # Expected [2, 16 * 16, 64]


def test_WindowLocalBlock():

    model = WindowLocalBlock(dim=64).eval()
    B, H, W, C = 2, 16, 16, 64
    N = H * W # 16 * 16 = 256

    x = torch.randn(B, N, C)
    out = model(x, H, W)
    print(out.shape) # Expected [2, 16 * 16, 64]

    # onnx testing
    torch.onnx.export(
        model, 
        (x, H, W), 
        "WindowLocalBlock.onnx", 
        input_names=["x"],
        output_names=["y"],
    )

    model = onnx.load("WindowLocalBlock.onnx")
    onnx.checker.check_model(model)

    sess = ort.InferenceSession("WindowLocalBlock.onnx")
    out = sess.run(
        None,
        {
            "x": x.numpy(),
        },
    )

    print(out[0].shape) # Expected [2, 16 * 16, 64]


def test_SGlobalBlock():

    model = SGlobalBlock(dim=64).eval()
    B, H, W, C = 2, 16, 16, 64
    N = H * W # 16 * 16 = 256

    x = torch.randn(B, N, C)
    out = model(x, H, W)
    print(out.shape) # Expected [2, 16 * 16, 64]

    # onnx testing
    torch.onnx.export(
        model, 
        (x, H, W), 
        "SGlobalBlock.onnx", 
        input_names=["x"],
        output_names=["y"],
    )

    model = onnx.load("SGlobalBlock.onnx")
    onnx.checker.check_model(model)

    sess = ort.InferenceSession("SGlobalBlock.onnx")
    out = sess.run(
        None,
        {
            "x": x.numpy(),
        },
    )

    print(out[0].shape) # Expected [2, 16 * 16, 64]


def test_TransformerModule():

    model = TransformerModule(dim=64).eval()
    B, H, W, C = 2, 16, 16, 64
    N = H * W # 16 * 16 = 256

    x = torch.randn(B, N, C)
    out = model(x, H, W)
    print(out.shape) # Expected [2, 16 * 16, 64]

    # onnx testing
    torch.onnx.export(
        model, 
        (x, H, W), 
        "TransformerModule.onnx", 
        input_names=["x"],
        output_names=["y"],
    )

    model = onnx.load("TransformerModule.onnx")
    onnx.checker.check_model(model)

    sess = ort.InferenceSession("TransformerModule.onnx")
    out = sess.run(
        None,
        {
            "x": x.numpy(),
        },
    )

    print(out[0].shape) # Expected [2, 16 * 16, 64]


def test_ConvStem():
    x = torch.randn(4, 3, 128, 128)
    conv_stem = ConvStem(in_channels=3, out_channels=64, num_layers=2)
    out = conv_stem(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    if out.shape == (4, 64, 128, 128):
        print("test_ConvStem: passed")
    else:
        print("test_ConvStem: failed")


def test_ConfFNN():
    x = torch.randn(4, 144, 144)
    conv_fnn= ConvFFN(dim=144)
    out = conv_fnn(x, 12, 12)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    if out.shape == (4, 144, 144):
        print("test_ConvStem: passed")
    else:
        print("test_ConvStem: failed")


def test_SpoofFormerNet():

    print("=" * 60)
    print("SpoofFormerNet – Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # --- Build model ---
    model = build_spoof_former_net("tiny").to(device)  # 'tiny' for fast test

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.2f} M")

    # --- Dummy inputs ---
    # B, H, W = 2, 256, 256
    B, H, W = 2, 256, 256

    rgb   = torch.randn(B, 3, H, W, device=device)
    depth = torch.randn(B, 1, H, W, device=device)

    # --- Forward pass ---
    model.eval()
    with torch.no_grad():
        logits = model(rgb, depth)
        probs  = model.predict(rgb, depth)

    print(f"Input  RGB   : {tuple(rgb.shape)}")
    print(f"Input  Depth : {tuple(depth.shape)}")
    print(f"Output logits: {tuple(logits.shape)}")
    print(f"Output probs : {tuple(probs.shape)}")
    print(f"Probs sum to 1: {probs.sum(dim=-1)}")

    # --- Loss ---
    # labels    = torch.randint(0, 2, (B,), device=device)
    # criterion = SpoofingLoss()
    # loss      = criterion(logits, labels)
    # print(f"Loss : {loss.item():.4f}")

    # # --- Metrics ---
    # real_scores = probs[:, 1]
    # metrics = compute_metrics(real_scores, labels)
    # print(f"Metrics: {metrics}")

    print("\nAll checks passed!")


test_ConvFFN()
# test_WeightedMSA()
# test_LocalWindowAttention()
# test_SparseGlobalAttention()
# test_WindowLocalBlock()
# test_SGlobalBlock()
# test_TransformerModule()

# test_ConvStem()
# test_ConfFNN()


if __name__ == "__main__":
 
    pass