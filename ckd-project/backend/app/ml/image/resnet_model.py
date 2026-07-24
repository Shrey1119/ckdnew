import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18KidneyClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet18KidneyClassifier, self).__init__()
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet18()
            
        # Keep references to features and classification head
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_classes)

    def extract_features(self, x):
        """
        Extract the 512-dimensional feature representation from avgpool layer
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, return_features=False):
        if return_features:
            features = self.extract_features(x)
            logits = self.resnet.fc(features)
            return logits, features
        return self.resnet(x)
