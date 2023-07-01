from captum.attr import GuidedGradCam
import torch
from resources.DLASimpleLoader import DLASimpleLoader
import os
from PIL import Image
import cv2
import numpy as np


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

# Visualize Guided Grad-Cam
heatmap = attribution.squeeze()
heatmap = heatmap.detach().numpy()
heatmap = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)

result = heatmap * 0.3 + input_image*0.5

print(attribution)