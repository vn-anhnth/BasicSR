import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ClassSRClassifier(nn.Module):
    """ClassSR Classifier (Class-Module) for License Plate Quality Assessment.

    Paper: ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic (CVPR 2021)
    Reference: https://github.com/Xiangtaokong/ClassSR

    Lightweight CNN architecture (~0.04M params) that categorizes incoming license plate patches into:
      - Class 0: Clear / High-Quality (Bypasses SR, directly to OCR)
      - Class 1: Degraded / Blur / Low-Quality (Routed to Super-Resolution before OCR)
    """
    def __init__(self, in_nc=3, num_classes=2, feat_dim=128):
        super(ClassSRClassifier, self).__init__()

        # Condition network from CVPR 2021 ClassSR
        self.CondNet = nn.Sequential(
            nn.Conv2d(in_nc, feat_dim, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_dim, 32, kernel_size=1, stride=1, padding=0)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.lastOut = nn.Linear(32, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is normalized in [0, 1]
        feat = self.CondNet(x)
        feat = self.global_pool(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.lastOut(feat)
        return out

# Command train ClassSR Classifier:
# powershell
# & "d:\IEEE\data\phD\OpenOCR\env\Scripts\python.exe" "d:\IEEE\data\phD\BasicSR\scripts\train_class_sr.py" --epochs 5
# model: d:\IEEE\data\phD\BasicSR\experiments\pretrained_models\class_sr_classifier_best.pth
