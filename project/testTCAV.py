import numpy as np
import os, glob

import matplotlib.pyplot as plt

from PIL import Image

#from scipy.stats import ttest_ind

# ..........torch imports............
import torch
import torchvision

from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

#.... Captum imports..................
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients

from captum.concept import TCAV
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

from resources.DLASimpleLoader import DLASimpleLoader


model_path = os.path.join(
"captumcv", "model_weights", "SimpleDLA_10epochs_cifar10.pth"
)
model_loader = DLASimpleLoader(model_path)



def get_tensor_from_filename(filename):
    img = Image.open(filename)
    return model_loader.preprocess_image(img)


def assemble_concept(name, id, concepts_path="resources/concepts/"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)

def load_image_tensors(class_name, root_path='resources/concepts/', transform=True):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.jpg')

    tensors = []
    for filename in filenames:
        img = Image.open(filename)
        tensors.append(model_loader.preprocess_image(img) if transform else img)

    return tensors

# Load sample images from folder
tiger_tensors = load_image_tensors('tiger', transform=True) #default False
#tiger_tensors = torch.stack([model_loader.preprocess_image(img) for img in tiger_imgs])
print(tiger_tensors.size())

concepts_path = "resources/concepts/"

honeycombed_concept = assemble_concept("honeycombed", 0, concepts_path=concepts_path)
striped_concept = assemble_concept("striped", 1, concepts_path=concepts_path)
random_1_concept = assemble_concept("random_1", 2, concepts_path=concepts_path)
random_2_concept = assemble_concept("random_2", 3, concepts_path=concepts_path)

layers=['layer3', 'layer4']

mytcav = TCAV(model=model_loader.model, layers=layers, layer_attr_method=LayerIntegratedGradients(model_loader.model, None, multiply_by_inputs=False))

experimental_set_rand = [[honeycombed_concept, random_1_concept], [striped_concept, random_2_concept]]

# tiger class index
tiger_ind = 1


tcav_scores_w_random = mytcav.interpret(inputs=tiger_tensors,
                                    experimental_sets=experimental_set_rand,
                                    target=tiger_ind,
                                    n_steps=5,
                                    )
print(tcav_scores_w_random)

def evaluate_button_TCAV(concepts, model, layers):
    pass