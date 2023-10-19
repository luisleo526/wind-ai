from collections import OrderedDict

import torch
from monai.networks.blocks import UpSample, Convolution
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork

from utils import get_class


class WindNet(torch.nn.Module):
    def __init__(self, cfg):
        super(WindNet, self).__init__()

        backbone = get_class(cfg.model.backbone.type)(**cfg.model.backbone.params)
        backbone.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3)

        self.features = create_feature_extractor(backbone, {
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        })
        self.fpn = FeaturePyramidNetwork(cfg.model.features_dims, cfg.model.fpn_channels)

        self.v1 = torch.nn.Sequential(OrderedDict([
            (f"layer_{layer_id}", get_class(layer.type)(**layer.params)) for layer_id, layer in
            enumerate(cfg.model.layers)
        ]))

        self.v2 = torch.nn.Sequential(OrderedDict([
            (f"layer_{layer_id}", get_class(layer.type)(**layer.params)) for layer_id, layer in
            enumerate(cfg.model.layers)
        ]))

        self.v3 = torch.nn.Sequential(OrderedDict([
            (f"layer_{layer_id}", get_class(layer.type)(**layer.params)) for layer_id, layer in
            enumerate(cfg.model.layers)
        ]))

        self.v2.append(UpSample(2, 1, 1))
        self.v3.append(UpSample(2, 1, 1))
        self.v3.append(UpSample(2, 1, 1))

        self.final = torch.nn.Sequential(
            Convolution(2, 3, 1, conv_only=False, norm="BATCH", act="SIGMOID"),
            UpSample(2, 1, 1),
            Convolution(2, 1, 1, conv_only=False, norm="BATCH", act="SIGMOID"),
            UpSample(2, 1, 1),
            Convolution(2, 1, 1, conv_only=False, norm="BATCH", act="SIGMOID"),
            UpSample(2, 1, 1),
            Convolution(2, 1, 1, conv_only=False, norm="BATCH", act="SIGMOID"),
        )

    def forward(self, x, scalar):
        # x : [B, C, H, W]
        # scalar: [B, N]

        batch_size, num_scalars = scalar.shape
        scalar = scalar.unsqueeze(-1).unsqueeze(-1)

        features = self.features(x)

        features = self.fpn(features)

        v1, v2, v3 = [
            torch.cat([feature, scalar.expand([batch_size, num_scalars, feature.shape[2], feature.shape[3]])], dim=1)
            for feature in features.values()]

        v1 = self.v1(v1)
        v2 = self.v2(v2)
        v3 = self.v3(v3)

        y = self.final(torch.cat([v1, v2, v3], dim=1))

        return y
