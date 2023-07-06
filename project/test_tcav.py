# ..........torch imports............
import torch
import torchvision

from torch.utils.data import IterableDataset, DataLoader

#.... Captum imports..................
from captum.attr import LayerIntegratedGradients

from captum.concept import TCAV
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

import numpy as np
import os, glob
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import ttest_ind
import zipfile

from resources.DLASimpleLoader import DLASimpleLoader

# Load model
model_path = os.path.join(
    "captumcv", "model_weights", "SimpleDLA_10epochs_cifar10.pth"
)
model_loader = DLASimpleLoader(model_path)
model = model_loader.model.module.to("cpu")

# Method to process image implemented in model_loader
def load_image_tensors_from_zip_file(zip_file, transforming=True):
    tensors = []
    for file_name in zip_file.namelist():
        with zip_file.open(file_name) as image_file:
            img = Image.open(image_file).convert('RGB')
            tensors.append(model_loader.preprocess_image(img) if transforming else img)
    return tensors

# assemble concept
def assemble_concept_from_zip(zip_file, name, id):
    image_tensors = load_image_tensors_from_zip_file(zip_file)
    return Concept(id=id, name=name, data_iter=dataset_to_dataloader(image_tensors))

concept_zip = zipfile.ZipFile("resources/striped.zip", "r")
concept_name = concept_zip.filename.split('.')[0]
random_zip = zipfile.ZipFile("resources/random_1.zip", "r")
random_name = random_zip.filename.split('.')[0]
class_zip = zipfile.ZipFile("resources/tiger.zip", "r")
class_name = class_zip.filename.split('.')[0]

# stripes and random
concept = assemble_concept_from_zip(concept_zip, name=concept_name, id=0)
random_concept = assemble_concept_from_zip(random_zip, name=random_name, id=1)

# tigers
class_tensors = load_image_tensors_from_zip_file(class_zip)

# layer of model
layers=['layer3']

mytcav = TCAV(model=model,
              layers=layers,
              layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))

experimental_set_rand = [[concept, random_concept]]

# tiger class index
tiger_ind = 293

# Problems with accepting tuple of tensors.
# workaround: loop through all tensors and calculate TCAV individually
# TODO: aggregate results
tcav_scores = []
for class_tensor in class_tensors:
    tcav_scores_w_random = mytcav.interpret(inputs=class_tensor.unsqueeze(0),
                                            experimental_sets=experimental_set_rand,
                                            target=tiger_ind,
                                            n_steps=5,
                                           )
    tcav_scores.append(tcav_scores_w_random)

print(tcav_scores)
