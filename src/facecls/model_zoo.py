import torch.nn as nn
from torchvision import models

def build_model(backbone: str, num_classes: int, freeze_backbone=False):
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return model
