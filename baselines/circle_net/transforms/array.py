import torch
import numpy as np
import torch
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

    def __call__(self, output: dict) -> dict:
        segmentation = torch.zeros(
            output[self.heatmap_key].shape
            ).to(output[self.heatmap_key].device)

        # Transform heatmap as numpy
        heatmap = output[self.heatmap_key].detach().cpu().numpy()
        radii = output[self.radius_key].detach().cpu().numpy()
        for i in range(segmentation.shape[0]):
            center_x, center_y = np.unravel_index(
                np.argmax(heatmap[i, 0]), heatmap[i, 0].shape
                )
            radius = radii[
                i, 0, center_x, center_y
                ]
            indices = disk((center_x, center_y), radius)
            segmentation[i, 0][indices] = 1
        return segmentation
