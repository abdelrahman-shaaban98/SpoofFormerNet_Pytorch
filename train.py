import time
import copy
from pathlib import Path
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from wandb_utils import log_model_wandb, log_train_metrics_wandb
from utils import count_params, save_model, print_training_progress
from config import CFG
from loss import SpoofingLoss
from metrics import compute_metrics
from dataloader import get_dataloaders
from model import SpoofFormerNet, build_spoof_former_net


def train_one_epoch(
    model: SpoofFormerNet,
    optimizer: torch.optim.Optimizer,
    loader,
    criterion: SpoofingLoss,
    device: torch.device,
    scaler=None,   # torch.cuda.amp.GradScaler for mixed precision
) -> dict:
    """
    Run one full training pass over 'loader', return a dict with:
        loss, accuracy
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, leave=False, desc="Training"):
        rgb, depth, labels = batch
        rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(rgb, depth)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(rgb, depth)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return dict(loss=(total_loss / total), accuracy=(correct / total))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: SpoofingLoss,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    """
    Run one full evaluation pass over 'loader', return a dict with:
        loss, accuracy, apcer, bpcer, acer, eer, auc
    """
    model.eval()
    total_loss = 0.0
    all_scores  = [] # score of 'real' class probability
    all_labels  = []

    for rgb, depth, labels in tqdm(loader, leave=False, desc="Validation"):
        rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(rgb, depth)
            loss   = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1)[:, 1]   # P(real)
        total_loss  += loss.item() * labels.size(0)
        all_scores.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_scores, all_labels)
    metrics["loss"] = total_loss / len(all_labels)

    return metrics


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def train(run, cfg: dict = CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["mixed_precision"] and device.type == "cuda"
    
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  SpoofFormerNet — Training")
    print("=" * 65)

    train_loader, test_loader = get_dataloaders(
        train_color_dir = cfg["train_color_dir"],
        train_depth_dir = cfg["train_depth_dir"],
        test_color_dir  = cfg["test_color_dir"],
        test_depth_dir  = cfg["test_depth_dir"],
        batch_size      = cfg["batch_size"],
        num_workers     = cfg["num_workers"],
        balance_classes = cfg["balance_classes"],
        verbose         = True,
    )

    model = build_spoof_former_net(cfg["model_variant"]).to(device)

    print(f"\n  Trainable params: {count_params(model)}\n")

    # Loss / optimiser / scaler 
    criterion = SpoofingLoss(label_smoothing=cfg["label_smoothing"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scaler = GradScaler() if use_amp else None

    #  State
    best_acer      = float("inf")
    best_weights   = None
    no_improve     = 0

    print(f"{'Ep':>4}  "
          f"{'Tr Loss':>8}  {'Tr Acc':>7}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  "
          f"{'APCER':>6}  {'BPCER':>6}  {'ACER':>6}  {'AUC':>7}  "
          f"{'ETA':>6}")
    print("-" * 100)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Train 
        train_metrics = train_one_epoch(
            model     = model,
            optimizer = optimizer,
            loader    = train_loader,
            criterion = criterion,
            device    = device,
            scaler    = scaler,
        )

        # Validate 
        val_metrics = evaluate(
            model    = model,
            loader   = test_loader,
            criterion= criterion,
            device   = device,
            use_amp  = use_amp,
        )


        # Time estimate 
        epoch_time  = time.time() - t0
        epochs_left = cfg["epochs"] - epoch
        eta         = _fmt_time(epoch_time * epochs_left)

        # Console output 
        improved = val_metrics["acer"] < best_acer - cfg["min_delta"]
        flag     = " ✓" if improved else ""

        print_training_progress(epoch, train_metrics, val_metrics, eta, flag)
        log_train_metrics_wandb(run, epoch, train_metrics, val_metrics)

        # Best checkpoint
        if improved:
            best_acer    = val_metrics["acer"]
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0


            save_path = save_dir / "best_model.pt"
            save_model(save_path, epoch, model, optimizer, cfg, val_metrics)

            log_model_wandb(run, save_dir / "best_model.pt",
                            "best_model")
            
        else:
            no_improve += 1

        # Periodic checkpoint 
        if epoch % cfg["save_every"] == 0:
            save_path = save_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            save_model(save_path, epoch, model, optimizer, cfg, val_metrics)

            log_model_wandb(run, save_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                            "periodic_checkpoint")
            
        #  Early stopping 
        if no_improve >= cfg["patience"]:
            print(f"\n  Early stopping — no improvement for {cfg['patience']} epochs.")
            break

    # Summary 
    print("\n" + "=" * 65)
    print("  Training complete.")
    print(f"  Best val ACER : {best_acer:.4f}")
    print(f"  Best model    : {save_dir / 'best_model.pt'}")
    print("=" * 65)



if __name__ == "__main__":
    run = wandb.init(
        entity="abdelrahman-shaaban-98",
        project="SpoofFormer_test",
        config=CFG,
    )

    train(run, CFG)
    run.finish()
