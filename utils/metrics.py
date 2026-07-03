"""Metrics utilities for OctFractalGen training and evaluation.

Contains:
  - get_correct_topk: top-k correctness check (used by VQ accuracy + remask)
  - compute_vq_accuracy: BSQ32 VQ prediction accuracy metrics
  - log_metric helpers: base-name extraction, focus filtering, sorting,
    formatting for training logs
"""
import torch


# ---------------------------------------------------------------------------
# VQ (BSQ32) accuracy
# ---------------------------------------------------------------------------
def get_correct_topk(logits, targets, topk=1):
    """Return a bool tensor indicating which top-k predictions are correct.

    Args:
        logits: (..., C) prediction logits
        targets: (...) ground-truth class indices
        topk: number of top predictions to consider. Clamped to C-1.
    Returns:
        bool tensor (..., topk) where True indicates a correct prediction.
    """
    topk = min(topk, logits.shape[-1] - 1)
    topk = torch.topk(logits, topk, dim=-1).indices
    return topk.eq(targets.unsqueeze(-1).expand_as(topk))


def compute_vq_accuracy(vq_logits, target_vq, mask=None,
                        vq_groups=32, vq_size=2, topk=5, device=None):
    """Compute VQ prediction accuracy metrics (no_grad context assumed).

    For BSQ32, vq_size=2 so topk=5 degenerates to topk=1 (argmax equality).

    Args:
        vq_logits: (N, vq_groups, vq_size) prediction logits
        target_vq: (N, vq_groups) ground-truth in {0, 1}
        mask: optional bool tensor (N,); if given, masked-only metrics are
            computed in addition to all-position metrics
        vq_groups: number of VQ groups (BSQ32 -> 32)
        vq_size: codebook size per group (BSQ32 -> 2)
        topk: top-k for vq_top5_acc (clamped to vq_size-1)
        device: torch device for the no-mask fallback zero tensor
    Returns:
        dict with keys: vq_bit_acc, vq_top5_acc, vq_bit_acc_all,
        vq_bit_acc_masked, vq_code_acc, vq_code_acc_all, vq_code_acc_masked,
        vq_mask_ratio
    """
    vq_pred_all = vq_logits.argmax(dim=-1)
    vq_bit_acc_all = (vq_pred_all == target_vq).float().mean()
    vq_code_acc_all = (vq_pred_all == target_vq).all(dim=-1).float().mean()
    vq_mask_ratio = (
        mask.float().mean()
        if mask is not None
        else torch.zeros((), device=device if device is not None else vq_logits.device)
    )

    if mask is not None:
        vq_pred_masked = vq_pred_all[mask]
        target_masked = target_vq[mask]
        vq_bit_acc_masked = (vq_pred_masked == target_masked).float().mean()
        vq_code_acc_masked = (
            vq_pred_masked == target_masked).all(dim=-1).float().mean()

        correct_top5 = get_correct_topk(
            vq_logits[mask].reshape(-1, vq_size),
            target_vq[mask].reshape(-1),
            topk=topk)
        vq_top5_acc = correct_top5.sum().float() / (
            mask.sum().float() * vq_groups).clamp_min(1.0)
    else:
        vq_bit_acc_masked = vq_bit_acc_all
        vq_code_acc_masked = vq_code_acc_all

        correct_top5 = get_correct_topk(
            vq_logits.reshape(-1, vq_size),
            target_vq.reshape(-1),
            topk=topk)
        vq_top5_acc = correct_top5.sum().float() / max(target_vq.numel(), 1)

    return {
        'vq_bit_acc': vq_bit_acc_masked,
        'vq_top5_acc': vq_top5_acc,
        'vq_bit_acc_all': vq_bit_acc_all,
        'vq_bit_acc_masked': vq_bit_acc_masked,
        'vq_code_acc': vq_code_acc_masked,
        'vq_code_acc_all': vq_code_acc_all,
        'vq_code_acc_masked': vq_code_acc_masked,
        'vq_mask_ratio': vq_mask_ratio,
    }


# ---------------------------------------------------------------------------
# Training-log metric helpers
# ---------------------------------------------------------------------------
def log_metric_base_name(key):
    """Strip the 'avg_' prefix used in epoch-averaged metric keys."""
    return key[4:] if key.startswith("avg_") else key


def is_focus_log_metric(key):
    """Return True for metrics that should appear in the focus log line."""
    base = log_metric_base_name(key)
    return base.startswith("split_acc_l") or base == "vq_top5_acc"


def focus_metric_sort_key(key):
    """Sort key: split_acc_l{N} first (by level), then vq_top5_acc, then others."""
    base = log_metric_base_name(key)
    if base.startswith("split_acc_l"):
        try:
            return (0, int(base[len("split_acc_l"):]))
        except ValueError:
            return (0, 99)
    if base == "vq_top5_acc":
        return (1, 0)
    return (2, base)


def format_focus_metrics(metrics):
    """Format focus metrics (split_acc_l*, vq_top5_acc) as a log string."""
    parts = []
    for key in sorted(metrics, key=focus_metric_sort_key):
        if not is_focus_log_metric(key):
            continue
        value = metrics[key]
        value = value.item() if torch.is_tensor(value) else float(value)
        parts.append(f"{key} {value:.3f}")
    return " ".join(parts)
