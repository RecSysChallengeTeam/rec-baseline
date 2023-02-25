import torch

from typing import Callable
from typing import Dict

from rec_baseline.common import get_device


class BaseTrainer:
    """
    BaseTrainer is a base class for all trainers (mostly neural network models).
    """
    
    # TODO: add save, load methods, and early stopping
    
    def __init__(self, model, optimizer: torch.optim.Optimizer, loss_fn: Callable, device: str = get_device()):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.model = self.model.to(self.device)
    
    def fit(self, train_loader, val_loader, epochs, verbose: int = 1) -> Dict:
        """
        Fit the model.
        
        Args:
            train_loader: training data loader
            val_loader: validation data loader
            epochs: number of epochs
            verbose: verbosity level
            
        Returns:
            history: training history
        """
        
        training_loss_list = []
        validation_list = []
                
        for epoch in range(epochs):
            
            training_loss_list.append(self.train_one_epoch(train_loader, epoch))
            validation_list.append(self.validate(val_loader, epoch))
            
            if verbose and epoch % verbose == 0:
                validation_format = ", ".join([f"{k}: {v:.4f}" for k, v in validation_list[-1].items()])
                print(f'Epoch: {epoch + 1}, training_loss: {training_loss_list[-1]:.4f}, {validation_format}')
        
        history = {k: [v[k] for v in validation_list] for k in validation_list[0]}
        history["training_loss"] = training_loss_list
        
        return history
            
    def train_one_epoch(self, train_loader, epoch):
        raise NotImplementedError

    def validate(self, dataloader, epoch):
        raise NotImplementedError
