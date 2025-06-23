import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet18 and modify last layer
model = models.resnet18()
model.fc = nn.Linear(512, 100)
model = model.to(device)
