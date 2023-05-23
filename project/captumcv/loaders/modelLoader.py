import torch
import torch.backends.cudnn as cudnn
import os
from captumcv.models.dla_simple import SimpleDLA


class ImageModelWrapper:
    """Wrapper base class for image models."""

    def __init__(self, image_shape, model_path, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # loaded model
        self.model = model
        self.model_path = model_path
        print(model_path)
        if model_path:
            if os.path.exists(model_path):
                self.__try_loading_model(model_path)
            else:
                raise Exception("Model path does not exist")
        # shape of the input image in this model
        self.image_shape = image_shape

    def predict(self, X: torch.Tensor):
        with torch.no_grad():
            output = self.model(X.to(self.device))
        return output

    def __try_loading_model(self, model_path: str):
        # prepare model
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

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


# example model definition
class DLASimpleLoader(ImageModelWrapper):
    def __init__(self, model_path):
        # TODO somehow automate the selection or import of SimpleDLA class ?
        model = SimpleDLA()
        image_shape = [1, 3, 32, 32]
        super(DLASimpleLoader, self).__init__(image_shape, model_path, model)
