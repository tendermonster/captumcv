import os
from typing import List

import torch
import torch.backends.cudnn as cudnn

class ImageModelWrapper:
    """Wrapper base class for image models."""

    def __init__(self, image_shape: List[int], model_path: str, model):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # loaded model
        self.model = model
        self.model_path = model_path
        print(model_path)

        
        if model_path and os.path.exists(model_path):
            self.__try_loading_model(model_path)
            #else:
                #raise Exception("Model path does not exist")
        # shape of the input image in this model
        self.image_shape = image_shape

    def predict(self, X: torch.Tensor):
        with torch.no_grad():
            output = self.model(X.to(self.device))
        return output

    def __try_loading_model(self, model_path: str):
        # prepare model
        #self.model = self.model[0]
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
