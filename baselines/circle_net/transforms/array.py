import torch
import numpy as np
from monai.transforms import Transform
from skimage.draw import disk


class CircleNetToSegmentation(Transform):
    def __init__(
        self,
        image_key='image',
        heatmap_key='heatmap',
        radius_key='radius',
    ):
        super().__init__()
        self.image_key = image_key
        self.heatmap_key = heatmap_key
        self.radius_key = radius_key
        self.relu = torch.nn.ReLU().float()
        self.sigmoid = torch.nn.Sigmoid().float()

    def __call__(self, output: dict) -> dict:
        segmentation = torch.zeros(
            output[self.heatmap_key].shape
            ).to(output[self.heatmap_key].device)

        # Transform heatmap as numpy
        heatmap = self.sigmoid(output[self.heatmap_key]).detach().cpu().numpy()
        radii = self.relu(output[self.radius_key]).detach().cpu().numpy()
        for i in range(segmentation.shape[0]):
            center_x, center_y = np.unravel_index(
                np.argmax(heatmap[i, 0]), heatmap[i, 0].shape
                )
            radius = radii[
                i, 0, center_x, center_y
                ]
            indices = disk((center_x, center_y), radius)
            # Filter indices to avoid out of range
            filtered_indices = (
                (indices[0] < segmentation.shape[-2]) *\
                (indices[1] < segmentation.shape[-1])
                )
            segmentation[i, 0][
                (
                    indices[0][filtered_indices],
                    indices[1][filtered_indices]
                    )
                ] = 1
        return segmentation
