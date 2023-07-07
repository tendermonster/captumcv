# example model definition
from typing import TYPE_CHECKING

from captumcv.loaders.modelLoader import ImageModelWrapper
from captumcv.tmp.model_def.dla_simple import SimpleDLA

# from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import PIL
    import torch
    import torchvision.transforms as transforms


class DLASimpleLoader(ImageModelWrapper):
    def __init__(self, model_path):
        # TODO somehow automate the selection or import of SimpleDLA class ?
        model = SimpleDLA()
        input_shape = [1, 3, 32, 32]
        target_image_shape = (32, 32)
        normalization_params = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        }
        super(DLASimpleLoader, self).__init__(
            input_shape, target_image_shape, model_path, model, normalization_params
        )

    # custom preprocess_image method can be defined here
    def preprocess_image(self, *args, **kwargs):
        return super().preprocess_image(*args, **kwargs)
