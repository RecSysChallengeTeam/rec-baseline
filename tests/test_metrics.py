import torch
import pytest

from rec_baseline.metrics import get_mrr, get_hit_ratio


def test_mrr():
    predicted_items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    mrr = get_mrr(predicted_items, target_item=torch.tensor([1, 2]))
    assert mrr == 0.75
    
    mrr = get_mrr(predicted_items, target_item=torch.tensor([1, 3]))
    assert pytest.approx(mrr, rel=1e-2) == 0.666
    
    mrr = get_mrr(predicted_items, target_item=torch.tensor([2, 3]), k=1)
    assert mrr == 0.0


def test_hit_ratio():
    predicted_items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    hr = get_hit_ratio(predicted_items, target_item=torch.tensor([1, 2]))
    assert hr == 1.0
    
    hr = get_hit_ratio(predicted_items, target_item=torch.tensor([1, 3]))
    assert hr == 1.0
    
    hr = get_hit_ratio(predicted_items, target_item=torch.tensor([2, 3]), k=1)
    assert hr == 0.0
