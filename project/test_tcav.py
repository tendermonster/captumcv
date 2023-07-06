# ..........torch imports............
import glob
import os
import zipfile
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# .... Captum imports..................
from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV, Concept
from captum.concept._utils.common import concepts_to_str
from captum.concept._utils.data_iterator import (
    CustomIterableDataset,
    dataset_to_dataloader,
)
from PIL import Image
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader, IterableDataset, Dataset

from resources.DLASimpleLoader import DLASimpleLoader

# Load model
model_path = os.path.join("captumcv", "model_weights", "SimpleDLA_10epochs_cifar10.pth")
model_loader = DLASimpleLoader(model_path)
model = model_loader.model.module.to("cpu")


# Method to process image implemented in model_loader
def load_image_tensors_from_zip_file(zip_file, transforming=True):
    tensors = []
    for file_name in zip_file.namelist():
        with zip_file.open(file_name) as image_file:
            img = Image.open(image_file).convert("RGB")
            tensors.append(model_loader.preprocess_image(img) if transforming else img)
    return tensors


# assemble concept
def assemble_concept_from_zip(zip_file, name, id):
    image_tensors = load_image_tensors_from_zip_file(zip_file)
    return Concept(id=id, name=name, data_iter=dataset_to_dataloader(image_tensors))


def load_image_tensors_from_paths(paths: List[str], transforming=True):
    tensors = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        tensors.append(model_loader.preprocess_image(img) if transforming else img)
    return tensors


# assemble concept
def assemble_concept_from_paths(paths, name, id):
    image_tensors = load_image_tensors_from_paths(paths)
    return Concept(
        id=id,
        name=name,
        data_iter=DataLoader(image_tensors, batch_size=None, batch_sampler=None),
    )


def format_float(f):
    return float("{:.3f}".format(f) if abs(f) >= 0.0005 else "{:.3e}".format(f))


def plot_tcav_scores(experimental_sets, tcav_scores):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize=(25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):
        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i - 1]])
        _ax = ax[idx_es] if len(experimental_sets) > 1 else ax
        for i in range(len(concepts)):
            val = [
                format_float(scores["sign_count"][i])
                for layer, scores in tcav_scores[concepts_key].items()
            ]
            _ax.bar(
                pos[i], val, width=barWidth, edgecolor="white", label=concepts[i].name
            )

        # Add xticks on the middle of the group bars
        _ax.set_xlabel("Set {}".format(str(idx_es)), fontweight="bold", fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    plt.show()


if __name__ == "__main__":
    # concept_zip = zipfile.ZipFile("resources/striped.zip", "r")
    # concept_name = concept_zip.filename.split(".")[0]
    # random_zip = zipfile.ZipFile("resources/random_1.zip", "r")
    # random_name = random_zip.filename.split(".")[0]
    # class_zip = zipfile.ZipFile("resources/tiger.zip", "r")
    # class_name = class_zip.filename.split(".")[0]

    # 1 zip hochladen
    # die in etwa so aussehen:
    # tcav.zip
    # | -> target_concept0
    # | -> random0
    # | -> random1
    # | -> random2
    # | -> target_class0
    # feld name the concept

    concept_striped = glob.glob("resources/concepts/striped/*")
    concept_random = glob.glob("resources/concepts/random_1/*")
    class_tcav = glob.glob("resources/concepts/tiger/*")
    # stripes and random
    concept = assemble_concept_from_paths(concept_striped, name="striped", id=0)
    random_concept = assemble_concept_from_paths(concept_random, name="random", id=2)
    concept1 = assemble_concept_from_paths(concept_striped, name="striped", id=1)
    random_concept1 = assemble_concept_from_paths(concept_random, name="random", id=3)
    print()

    # tigers
    class_tensors = load_image_tensors_from_paths(class_tcav)

    # layer of model
    layers = ["layer3"]

    mytcav = TCAV(
        model=model,
        layers=layers,
        layer_attr_method=LayerIntegratedGradients(
            model, None, multiply_by_inputs=False
        ),
    )

    experimental_set_rand = [[concept, random_concept], [concept1, random_concept1]]

    # tiger class index
    tiger_ind = 2

    # Problems with accepting tuple of tensors.
    # workaround: loop through all tensors and calculate TCAV individually
    # TODO: aggregate results
    tcav_scores = {}
    for class_tensor in class_tensors:
        tcav_scores_w_random = mytcav.interpret(
            inputs=class_tensor,
            experimental_sets=experimental_set_rand,
            target=tiger_ind,
            n_steps=5,
        )
        tcav_scores.update(tcav_scores_w_random)

    plot_tcav_scores(experimental_sets=experimental_set_rand, tcav_scores=tcav_scores)
