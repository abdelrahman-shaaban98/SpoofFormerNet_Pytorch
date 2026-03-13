from sklearn.metrics import roc_auc_score
import torch


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    AUC computation [Eq. 13].
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
        scores: probability of being real (class 1)
        labels: ground truth  (0=fake, 1=real)
        threshold: classification threshold

    Returns dict with keys: accuracy, eer, apcer, bpcer, acer, auc
    """
    preds = (scores >= threshold).long()
    accuracy = (preds == labels).float().mean().item()

    real_mask = labels == 1
    fake_mask = labels == 0

    # EER: threshold where FAR ≈ FRR  [Eq. 14]
    thresholds = torch.linspace(0, 1, 200, device=scores.device)
    best_eer = 1.0
    for t in thresholds:
        preds = (scores >= t).long()
        far = (preds[fake_mask] == 1).float().mean().item() if fake_mask.any() else 0.0 # False acceptance rate
        frr = (preds[real_mask] == 0).float().mean().item() if real_mask.any() else 0.0 # Falce rejection rate
        eer_t = abs(far - frr)
        if eer_t < best_eer:
            best_eer = eer_t
            eer = (far + frr) / 2.0

    # APCER: spoof samples wrongly classified as real  [Eq. 15]
    apcer = (preds[fake_mask] == 1).float().mean().item() if fake_mask.any() else 0.0 # Avoid zero division

    # BPCER: real samples wrongly classified as spoof  [Eq. 16]
    bpcer = (preds[real_mask] == 0).float().mean().item() if real_mask.any() else 0.0 # Avoid zero division

    # ACER  [Eq. 17]
    acer  = (apcer + bpcer) / 2.0

    auc = compute_auc(scores, labels)

    return dict(accuracy=accuracy, eer=eer, apcer=apcer, bpcer=bpcer, acer=acer, auc=auc)

