import torch


def get_mrr(
    predicted_items: torch.Tensor,
    target_item: torch.Tensor,
    k=10,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate the Mean Reciprocal Rank (MRR) metric.

    Args:
        predicted_items: The predictions, (batch_size, n_items).
        target_item: The ground truth, (batch_size, )
        k: The number of top items to consider.
        
    Returns:
        MRR score
    """

    top_hit = (predicted_items[:, :k] == target_item.unsqueeze(1))
    div = torch.arange(1, k + 1).to(predicted_items.device)
    mrr = (top_hit / div).sum(dim=-1)
    mrr = mrr.detach().cpu().numpy()
    if reduction == "mean":
        return mrr.mean()
    elif reduction == "sum":
        return mrr.sum()
    return mrr


def get_hit_ratio(
    predicted_items: torch.Tensor,
    target_item: torch.Tensor,
    k: int = 10,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate the hit ratio metric.

    Args:
        predicted_items: The predictions, (batch_size, n_items).
        target_item: The ground truth, (batch_size, )
        k: The number of top items to consider.
        
    Returns:
        hit ratio score
    """

    top_hit = (predicted_items[:, :k] == target_item.unsqueeze(1))
    hr = top_hit.sum(dim=-1)
    hr = hr.detach().cpu().numpy()
    if reduction == "mean":
        return hr.mean()
    elif reduction == "sum":
        return hr.sum()
    return hr
