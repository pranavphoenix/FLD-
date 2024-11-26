import torch
import torch.nn as nn
import torchvision.models as models
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn.nets import ResidualNet

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        resnet = models.resnet18(pretrained=True)
        # Remove the fully connected layer and global pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: 512x8x8
        self.pool = nn.AvgPool2d(2, stride=2)
        self.pool1 = nn.AvgPool2d(2, stride=2)

        # Freeze the ResNet weights
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool1(x)
        return  self.pool(x) # Output shape: (batch_size, 512, 4, 4)
    

class SplineFlow(nn.Module):
    def __init__(self, in_channels=512 * 2 * 2, num_blocks=6, hidden_channels=128, num_bins=4):
        super(SplineFlow, self).__init__()
        transforms = []

        torch.manual_seed(42)

        # Create a sequence of coupling layers with splines
        for _ in range(num_blocks):
            transforms.append(ReversePermutation(features=in_channels))  # Reverse permutation for mixing channels
            transforms.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=nn.Parameter(torch.randn(in_channels) > 0.5, requires_grad=False),
                    transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        hidden_features=hidden_channels,
                        context_features=None,
                        num_blocks=2,
                    ),
                    num_bins=num_bins,
                    tails='linear',
                    tail_bound=3.0
                )
            )

        # Combine the transforms into a flow
        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([in_channels])
        )

    def forward(self, x):
        return self.flow.log_prob(x)

# class Backbone(nn.Module):
#     def __init__(self):
#         super(Backbone, self).__init__()

#         mobilenet = models.mobilenet_v3_small(pretrained=True)
#         # Remove the fully connected layer and global pooling
#         self.backbone = nn.Sequential(*list(mobilenet.children())[:-2])  # Output: 512x8x8
#         self.pool1 = nn.AvgPool2d(2, stride=2)
#         self.pool2 = nn.AvgPool2d(2, stride=2)

#         # Freeze the ResNet weights
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.pool1(x)
#         # print(x.shape)
#         return  self.pool2(x) # Output shape: (batch_size, 512, 4, 4)
    

# class SplineFlow(nn.Module):
#     def __init__(self, in_channels=576 * 2 * 2, num_blocks=6, hidden_channels=128, num_bins=4):
#         super(SplineFlow, self).__init__()
#         transforms = []

#         torch.manual_seed(42)

#         # Create a sequence of coupling layers with splines
#         for _ in range(num_blocks):
#             transforms.append(ReversePermutation(features=in_channels))  # Reverse permutation for mixing channels
#             transforms.append(
#                 PiecewiseRationalQuadraticCouplingTransform(
#                     mask=nn.Parameter(torch.randn(in_channels) > 0.5, requires_grad=False),
#                     transform_net_create_fn=lambda in_features, out_features: ResidualNet(
#                         in_features=in_features,
#                         out_features=out_features,
#                         hidden_features=hidden_channels,
#                         context_features=None,
#                         num_blocks=2,
#                     ),
#                     num_bins=num_bins,
#                     tails='linear',
#                     tail_bound=3.0
#                 )
#             )

#         # Combine the transforms into a flow
#         self.flow = Flow(
#             transform=CompositeTransform(transforms),
#             distribution=StandardNormal([in_channels])
#         )

#     def forward(self, x):
#         return self.flow.log_prob(x)


class ImageFlow(nn.Module):
    def __init__(self):
        super(ImageFlow, self).__init__()

        self.backbone = Backbone()
        self.flow = SplineFlow()

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return  self.flow(x)