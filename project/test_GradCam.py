from captum.attr import GuidedGradCam
import torch
from resources.DLASimpleLoader import DLASimpleLoader
import os
from PIL import Image


model_path = os.path.join(
    "captumcv", "model_weights", "SimpleDLA_10epochs_cifar10.pth"
)
img = Image.open(
    os.path.join("resources", "testbild.jpg")
)  

model_loader = DLASimpleLoader(model_path)
layer_name = ... # only layers with single tensor output supported
guided_gc = GuidedGradCam(model_loader.model, layer_name)

input_image = model_loader.preprocess_image(image = img)# tensor
attribution = guided_gc.attribute(input_image, 3)

print(attribution)