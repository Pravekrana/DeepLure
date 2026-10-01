import torch


def supervised_contrastive(z, labels, temperature=.1):
    """Mean log-probability of all positive views; exclude self-pairs.

    In instance mode labels are original groups, NOT asserted design identities.
    Anchors with no positive are excluded; an entirely invalid batch is an error.
    """
    if len(z) < 2:
        raise ValueError('Contrastive loss requires at least two views')
    logits = z @ z.T / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid = positives.sum(1) > 0
    if not valid.any():
        raise ValueError('Batch has no valid positive pairs')
    log_denom = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1)
    log_prob = logits - log_denom[:, None]
    losses = -(log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1).clamp_min(1))
    return losses[valid].mean()
