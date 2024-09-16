import torch
from torch.nn.utils import prune

def prune_model(model, amount=0.3):
    """
    Apply weight pruning to a model.
    
    Args:
        model (torch.nn.Module): The model to prune.
        amount (float): The amount of weights to prune (default: 0.3).
    
    Returns:
        torch.nn.Module: The pruned model.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model
