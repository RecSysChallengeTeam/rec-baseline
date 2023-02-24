import torch

from typing import Callable, List
from cf_baseline.common import get_device
from cf_baseline.trainer.base import BaseTrainer


class MFTrainer(BaseTrainer):
    """
    Trainer for matrix factorization model.
    """
    
    def __init__(self, model, optimizer: torch.optim.Optimizer, loss_fn: Callable, device: str = get_device()):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.model = self.model.to(self.device)

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> List:
        loss_list = []
        
        self.model.train()
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
            user_ids, item_ids, ratings = user_ids.to(self.device), item_ids.to(self.device), ratings.to(self.device)
            preds = self.model(user_ids, item_ids)
            loss = self.loss_fn(preds, ratings)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_list.append(loss.item())
            
        return loss_list
            
    def validate(self, val_loader: torch.utils.data.DataLoader, epoch: int) -> List:
        loss_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (user_ids, item_ids, ratings) in enumerate(val_loader):
                user_ids, item_ids, ratings = user_ids.to(self.device), item_ids.to(self.device), ratings.to(self.device)
                preds = self.model(user_ids, item_ids)
                loss = self.loss_fn(preds, ratings)
                
                loss_list.append(loss.item())
                
        return loss_list