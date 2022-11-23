import torch


def calculate_mrr(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate the Mean Reciprocal Rank (MRR) metric.

    Args:
        predictions: The predictions, (batch_size, n_items).
        targets: The targets, (batch_size, n_items).
        reduction: The reduction method. Defaults to "mean".

    Returns:
        The MRR metric.
    """
    
    # Get the ranks of the targets
    ranks = (targets == predictions[:, 0].unsqueeze(1)).nonzero()[:, 1]

    # Calculate the MRR
    mrr = 1 / (ranks + 1).float()

    # Reduce the MRR
    if reduction == "mean":
        mrr = mrr.mean()
    elif reduction == "sum":
        mrr = mrr.sum()

    return mrr
