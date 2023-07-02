# example model definition
from typing import TYPE_CHECKING

# this custom loader need to import the ImageModelWrapper class
from captumcv.loaders.util.modelLoader import ImageModelWrapper

# aswell as the model itself
from captumcv.models.resnet import ResNet18

# do not worry if imports are not found. They will work as soon as
# file files will be uploaded via frontend

# from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import PIL
    import torch
    import torchvision.transforms as transforms


class ResNet18Loader(ImageModelWrapper):
    def __init__(self, model_path):
        model = ResNet18()  # define model here -> the variable name must be "model"
        input_shape = [
            1,
            3,
            32,
            32,
        ]  # define input shape here -> the variable name must be "input_shape"
        target_image_shape = (
            32,
            32,
        )  # define target image shape here -> the variable name must be "target_image_shape"
        normalization_params = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        }  # define normalization params here -> the variable name must be "normalization_params"
        super(ResNet18Loader, self).__init__(
            input_shape, target_image_shape, model_path, model, normalization_params
        )

    # custom preprocess_image method can be defined here
    def preprocess_image(self, *args, **kwargs):
        return super().preprocess_image(*args, **kwargs)
