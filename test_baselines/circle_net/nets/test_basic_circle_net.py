import numpy as np
import torch
from baselines.circle_net.nets import BasicCircleNet


def test_basic_circle_net():
    """Test basic circle net forward
    """
    circle_net = BasicCircleNet(
        spatial_dims=2,
        in_channels=5,
        out_channels_heatmap=1,
        out_channels_radius=3,
        out_channels_offset=2,
        )
    tensor = torch.from_numpy(
        np.arange(3 * 5 * 32 * 32).reshape(
            3, 5, 32, 32
            )
        ).float()
    outputs = circle_net(tensor)
    assert outputs['radius'].shape == (3, 3, 32, 32)
    assert outputs['heatmap'].shape == (3, 1, 32, 32)
