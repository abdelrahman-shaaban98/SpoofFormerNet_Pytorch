from sklearn.metrics import roc_auc_score
import torch


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Trapezoidal AUC computation from the ROC curve  [Eq. 13].
    """
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
        
    return float(roc_auc_score(labels_np, scores_np))


def compute_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Compute face anti-spoofing evaluation metrics.

    Args:
        scores : B – probability of being real (class 1)
        labels : B – ground truth  (0=spoof, 1=real)
        threshold: classification threshold

    Returns dict with keys: apcer, bpcer, acer, eer, accuracy
    """
    preds = (scores >= threshold).long()

    real_mask = labels == 1
    spoof_mask   = labels == 0

    # BPCER: real samples wrongly classified as spoof  [Eq. 16]
    bpcer = (preds[real_mask] == 0).float().mean().item() if real_mask.any() else 0.0
    # APCER: spoof samples wrongly classified as real  [Eq. 15]
    apcer = (preds[spoof_mask]   == 1).float().mean().item() if spoof_mask.any()   else 0.0
    # ACER  [Eq. 17]
    acer  = (apcer + bpcer) / 2.0

    accuracy = (preds == labels).float().mean().item()

    # EER: threshold where FAR ≈ FRR  [Eq. 14]
    thresholds = torch.linspace(0, 1, 200, device=scores.device)
    best_eer = 1.0
    for t in thresholds:
        p = (scores >= t).long()
        far = (p[spoof_mask]   == 1).float().mean().item() if spoof_mask.any()   else 0.0
        frr = (p[real_mask] == 0).float().mean().item() if real_mask.any() else 0.0
        eer_t = abs(far - frr)
        if eer_t < best_eer:
            best_eer = eer_t
            eer_val  = (far + frr) / 2.0

    auc = compute_auc(scores, labels)

    return dict(accuracy=accuracy, apcer=apcer, bpcer=bpcer, acer=acer, eer=eer_val, auc=auc)

