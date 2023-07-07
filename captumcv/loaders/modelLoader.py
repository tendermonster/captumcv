import os
from typing import List

import PIL
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


class ImageModelWrapper(object):
    """
    Wrapper base class for image models.

    Args:
        input_shape: Shape of the input image.
        target_image_shape (List[int]): Target shape of the image.
        model_path (str): Path to the model file.
        model: The model object.
        normalization_params: Normalization parameters for image preprocessing.
    """

    def __init__(
        self,
        input_shape,
        target_image_shape: List[int],
        model_path: str,
        model,
        normalization_params,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # loaded model
        self.input_shape = input_shape
        self.target_image_shape = target_image_shape
        self.model = model
        self.model_path = model_path
        self.normalization_params = normalization_params
        if model_path and os.path.exists(model_path):
            self.__try_loading_model(model_path)
            # else:
            # raise Exception("Model path does not exist")
        # shape of the input image in this model

    def change_device(self, device: str):
        """
        Changes the device on which the model is loaded.

        Args:
            device (str): Device name ('cpu' or 'cuda').

        Returns:
            None
        """
        # TODO add all check so that model changes device successfully
        if self.device == device:
            return
        self.device = device
        self.model.to(device)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs prediction on the input tensor.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output tensor.
        """
        with torch.no_grad():
            output = self.model(X.to(self.device))
        return output

    def preprocess_image(self, *args, **kwargs) -> torch.Tensor:
        """
        Transforms image to the right input size of the model
        needs some adjustment for dynamic variables, for custom transformations.
        This method can be overridden if any custom preprocessing is needed.

        Args:
            args[0]: Pillow Image or
            kwargs["image"]: Pillow Image or
        Returns:
            torch.Tensor: _description_
        """
        if "image" in kwargs:
            image = kwargs["image"]
        elif args:
            image = args[0]
        else:
            raise Exception("No image argument provided")
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.target_image_shape),  # in case of cifar10
                transforms.Normalize(
                    mean=self.normalization_params["mean"],
                    std=self.normalization_params["std"],
                ),
            ]
        )
        transformed_image = transform_test(image)
        reshaped_image = torch.reshape(transformed_image, self.get_input_shape())
        return reshaped_image

    def __try_loading_model(self, model_path: str):
        """
        Tries to load the model from the provided model path.

        Args:
            model_path (str): Path to the model file.

        Returns:
            None
        """
        self.model = self.model.to(self.device)
        try:
            # todo do some checks if file exists or so
            if self.device == "cuda":
                self.model = torch.nn.DataParallel(self.model)
                cudnn.benchmark = True
                model_load = torch.load(model_path)
            else:
                # see this https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
                self.model = torch.nn.DataParallel(self.model)
                model_load = torch.load(
                    model_path, map_location=torch.device(self.device)
                )
            self.model.load_state_dict(model_load["net"])
            self.model.eval()
        except Exception as e:
            print(e)

    def get_input_shape(self):
        """
        Returns the shape of an input image.

        Returns:
            Tuple[int]: Input image shape.
        """
        return self.input_shape
