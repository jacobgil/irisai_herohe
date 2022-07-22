import sys
import os
from torchvision import models, transforms
import torch

class Resnet50Classifier(torch.nn.Module):
    def __init__(self, num_categories=2):
        super(Resnet50Classifier, self).__init__()
        self.features = self.get_resnet_model_classifier()
        self.fc = torch.nn.Linear(2048, num_categories)

    def forward(self, x):
        f = self.features(x)
        return self.fc(f.view(x.size(0), -1))

    def get_resnet_model_classifier(self, model_name="resnet50", layers_to_drop=1):
        model = eval("models."+model_name+"(pretrained=True)")
        if layers_to_drop > 0:
            modules=list(model.children())[:-layers_to_drop]

            pooling_index = -1

            for i, m in enumerate(modules):
                if isinstance(m, torch.nn.AvgPool2d):
                    pooling_index = i
            if pooling_index != -1:
                modules[pooling_index] = torch.nn.AdaptiveAvgPool2d((1, 1))

            model = torch.nn.Sequential(*modules)
        return model
