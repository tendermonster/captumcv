from captum.attr import GuidedGradCam
import torch
from resources.DLASimpleLoader import DLASimpleLoader
import os
from PIL import Image
import cv2
import numpy as np
from captum.attr import visualization as viz
from matplotlib import pyplot as plt


model_path = os.path.join(
    "captumcv", "model_weights", "SimpleDLA_10epochs_cifar10.pth"
)

img = Image.open(
    os.path.join("resources", "testbild.jpg")
)  

model_loader = DLASimpleLoader(model_path)
model = model_loader.model.module.to("cpu")
layer_name = model.layer3  # only layers with single tensor output supported
guided_gc = GuidedGradCam(model_loader.model, layer_name)

input_image = model_loader.preprocess_image(image=img)  # tensor
attribution = guided_gc.attribute(input_image, 1)
attribution_np = np.transpose(
    attribution.squeeze().detach().cpu().numpy(), axes=(1, 2, 0)
)
img = np.array(img)
print(attribution.shape)

# Visualize Guided Grad-Cam
#heatmap = attribution.squeeze().detach().numpy().to
#heatmap = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
#result = heatmap*0.3 + input_image*0.5

f, ax = viz.visualize_image_attr_multiple(
    attribution_np,
    img,
    ["original_image", "heat_map"],
    ["all", "positive"],
    show_colorbar=True,
    outlier_perc=2,
)

plt.show(f)
print(attribution)