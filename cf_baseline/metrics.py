import torch


def calculate_mrr(
    predicted_items: torch.Tensor,
    target_item: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate the Mean Reciprocal Rank (MRR) metric.

    Args:
        predicted_items: The predictions, (batch_size, n_items).
        target_item: The ground truth, (batch_size, )
        
    Returns:
        MRR score
    """

    rank = (predicted_items == target_item.unsqueeze(1)).nonzero()[:, 1] + 1
    mrr = 1.0 / rank.float()

    if reduction == "mean":
        return mrr.mean()
    elif reduction == "sum":
        return mrr.sum()

    return mrr
