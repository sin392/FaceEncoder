# flake8: noqa

import torch.nn as nn
import torchvision.models as models


class my_vgg16_bn(nn.Module):
    def __init__(self, middle_features=2048, out_features=1000, pretrained=True):
        super().__init__()
        self.model = models.vgg16_bn(pretrained=pretrained)
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_features=in_features,
                          out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096,
                          out_features=middle_features, bias=True),
            )
        )
        self.last = nn.Sequential(
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=middle_features,
                          out_features=out_features, bias=True),
                # nn.Softmax(dim=1) # pytochにおいてsoftmaxは不要
            )
        )

    def forward(self, x):
        middle = self.model(x)
        out = self.last(middle)
        return middle, out
