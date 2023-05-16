# just a placeholder for python
import torch

class Loader:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
        return output