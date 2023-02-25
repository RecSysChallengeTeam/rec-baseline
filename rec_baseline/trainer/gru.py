import torch
import numpy as np

from typing import List
from rec_baseline.trainer.base import BaseTrainer
from rec_baseline.metrics import get_hit_ratio, get_mrr


class GRU4RecTrainer(BaseTrainer):
    """
    Trainer for GRU4Rec.
    """

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> List:
        loss_list = []
        
        self.model.train()
        for batch_idx, (views, purchase) in enumerate(train_loader):
            views, purchase = views.to(self.device), purchase.to(self.device)
            preds = self.model(views)
            loss = self.loss_fn(preds, purchase)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_list.append(loss.item())
            
        return np.mean(loss_list)
            
    def validate(self, val_loader: torch.utils.data.DataLoader, epoch: int) -> List:
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (views, purchase) in enumerate(val_loader):
                views, purchase = views.to(self.device), purchase.to(self.device)
                preds = self.model(views)
                loss = self.loss_fn(preds, purchase)
                
                metrics = {
                    "validation_loss": loss.item(),
                    "hit_ratio": get_hit_ratio(self.model.predict(views), purchase),
                    "mrr": get_mrr(self.model.predict(views), purchase),
                }
                
        return metrics
