import torch

from typing import Callable
from typing import Any, Tuple

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
    
    def fit(self, train_loader, val_loader, epochs, verbose: int = 1) -> Tuple:
        """
        Fit the model.
        
        Args:
            train_loader: training data loader
            val_loader: validation data loader
            epochs: number of epochs
            verbose: verbosity level
            
        Returns:
            train_losses: list of training losses
            val_losses: list of validation losses
        """
        
        training_loss_list = []
        validation_loss_list = []
                
        for epoch in range(epochs):
            
            training_loss_list.extend(self.train_one_epoch(train_loader, epoch))
            validation_loss_list.extend(self.validate(val_loader, epoch))
            
            if verbose and epoch % verbose == 0:
                print(f'Epoch: {epoch + 1}, Training Loss: {training_loss_list[-1]:.4f}, Validation Loss: {validation_loss_list[-1]:.4f}')
        
        return training_loss_list, validation_loss_list
            
    def train_one_epoch(self, train_loader, epoch):
        raise NotImplementedError

    def validate(self, val_loader, epoch):
        raise NotImplementedError
