import torch


def calculate_mrr(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate the Mean Reciprocal Rank (MRR) metric.

    Args:
        predictions (torch.Tensor): The predictions.
        targets (torch.Tensor): The targets.
        k (int, optional): The number of top predictions to consider. Defaults to 10.
        reduction (str, optional): The reduction method. Defaults to "mean".

    Returns:
        The MRR metric.
    """
    
    # Get the top k predictions
    top_k = predictions.topk(k=k, dim=1)[1]

    # Get the ranks of the targets
    ranks = torch.zeros_like(targets)
    for i in range(targets.shape[0]):
        ranks[i] = torch.where(top_k[i] == targets[i])[0]

    # Calculate the MRR
    mrr = 1 / (ranks + 1)

    # Reduce the MRR
    if reduction == "mean":
        mrr = mrr.mean()
    elif reduction == "sum":
        mrr = mrr.sum()

    return mrr
