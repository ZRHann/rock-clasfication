import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=9):
    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),    # output [batch, 1536, 1, 1]
        nn.Flatten(1),              # flatten to [batch, 1536]
        nn.LayerNorm(1536),         # normalize
        nn.Dropout(p=0.2),          # dropout for regularization
        nn.Linear(1536, 768),
        nn.GELU(),
        nn.Dropout(p=0.4),          # dropout for regularization
        nn.Linear(768, num_classes)
    )

    # Freeze feature extraction layers
    for name, param in model.named_parameters():
        if "classifier" not in name:  # freeze all except classifier
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    return model

def get_model_name():
    return "convnext_large" 