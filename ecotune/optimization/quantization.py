import torch

def quantize_model(model):
    """
    Apply dynamic quantization to a model.
    
    Args:
        model (torch.nn.Module): The model to quantize.
    
    Returns:
        torch.nn.Module: The quantized model.
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
