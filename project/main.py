import os
import typing
from enum import Enum
from typing import Optional, Tuple

import joblib
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from captum.attr import (
    GradientShap,
    IntegratedGradients,
    LayerGradientXActivation,
    LayerIntegratedGradients,
    NeuronConductance,
    NoiseTunnel,
    Occlusion,
    Saliency,
)
from captum.attr import visualization as viz
from captum.concept import TCAV, Concept
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from captumcv.loaders.util.classLoader import (
    get_class_names_from_file,
    load_class_from_file,
)
from captumcv.loaders.util.modelLoader import ImageModelWrapper

if typing.TYPE_CHECKING:
    import matplotlib


class Attr(Enum):
    IG = "Integrated gradients"
    SALIENCY = "Saliency"
    TCAV = "TCAV"
    GRADCAM = "GradCam"
    NEURON_CONDUCTANCE = "Neuron Conductance"
    NEURON_GUIDED_BACKPROPAGATION = "Neuron Guided Backpropagation"
    DECONVOLUTION = "Deconvolution"

choose_method = st.selectbox(
    "Choose Attribution Method",
    (
        Attr.IG.value,
        Attr.SALIENCY.value,
        Attr.TCAV.value,
        Attr.GRADCAM.value,
        Attr.NEURON_CONDUCTANCE.value,
        Attr.NEURON_GUIDED_BACKPROPAGATION.value,
        Attr.DECONVOLUTION.value,
    ),
)
st.write("You selected:", choose_method)


file_cache = {}


## Modell  und Parameter auswählen
def parameter_selection():
    if choose_method == Attr.IG.value:
        options = [
            "Gausslegendre",
            "Riemann_left",
            "Riemann_right",
            "Riemann_middle",
            "Riemann_trapezoid",
        ]
        st.sidebar.selectbox("method:", options)
        if options == "Gausslegendre":
            st.write("aaa")
            st.sidebar.write("you choose Gausslegendre as parameter")
        elif options == "Riemann_left":
            st.sidebar.write("you choose Riemann_left as parameter")
        elif options == "Riemann_right":
            st.sidebar.write("you choose Riemann_right as parameter")
        elif options == "Riemann_middle":
            st.sidebar.write("you choose Riemann_middle as parameter")
        elif options == "Riemann_trapezoid":
            st.sidebar.write("you choose Riemann_trapezoid as parameter")
        st.sidebar.number_input("Insert step:", min_value=25, step=1)
    if choose_method == Attr.SALIENCY.value:
        st.sidebar.text("without parameter")
    if choose_method == Attr.TCAV.value:
        # need parameter from TCAV
        st.sidebar.write("you choose TCAV")
    if choose_method == Attr.GRADCAM.value:
        # need parameter from GradCam
        st.sidebar.write("you choose GradCam")
    if choose_method == Attr.NEURON_CONDUCTANCE.value:
        options = [
            "param1",
            "param2",
        ]
        st.sidebar.selectbox("method:", options)
        if options == "param1":
            st.write("aaa")
            st.sidebar.write("you choose param1 as parameter")
        elif options == "param2":
            st.sidebar.write("you choose param2 as parameter")
        st.sidebar.number_input("Layer:", min_value=0, step=1)
    if choose_method == Attr.NEURON_GUIDED_BACKPROPAGATION.value:
        st.sidebar.text("without parameter")
    if choose_method == Attr.DECONVOLUTION.value:
        st.sidebar.text("without parameter")

def __load_model(model_path: str, loader_class_name: str, model_loader_path: str) -> ImageModelWrapper:
    """
    This method loads the model from the given path and returns it.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path

    Returns:
        model|None: returns the loaded model or None if the model could not be loaded
    """
    model_loader = load_class_from_file(model_loader_path, loader_class_name)
    # check that the class extends correct subclass
    if model_loader and issubclass(model_loader, ImageModelWrapper):
        instance: ImageModelWrapper = model_loader(model_path)
        if isinstance(instance.model, torch.nn.DataParallel):
            model = instance.model.module.to('cpu') # if DataParallel do this to make it work on cpu
        else:
            model = instance.model
        return model
    return None

def __plot(true_img, attr_img) -> matplotlib.figure.Figure:
    # the original image should have the (H,W,C) format
    attr_img = np.flip(
        attr_img, axis=1
    )  # flip the image on y axis # BUG why is it even flipped ???
    f, ax = viz.visualize_image_attr_multiple(
        attr_img,
        true_img,
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
    )
    return f

# Function for IG
def evaluate_button_ig(
    input_image_path: str, model_path: str, loader_class_name: str, model_loader_path
):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """
    m = __load_model(model_path, loader_class_name, model_loader_path)
    if m is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    ig = IntegratedGradients(m.model)
    img = Image.open(input_image_path)
    img = np.array(img)  # convert to numpy array
    X_img = m.preprocess_image(img)
    attribution = ig.attribute(X_img, target=0)
    attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), (1, 2, 0))
    f = __plot(img, attribution_np)
    st.pyplot(f)
    st.write("Evaluation finished")

def evaluate_button_n_conductance(
    input_image_path: str,
    model_path: str,
    loader_class_name: str,
    model_loader_path: str,
):
    """
    This method runs the captum algorithm and shows the results.
    Use with DataParallel https://captum.ai/tutorials/Distributed_Attribution 

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """
    m = __load_model(model_path, loader_class_name, model_loader_path)
    if m is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    img = Image.open(input_image_path)
    img = np.array(img)  # convert to numpy array
    X_img = m.preprocess_image(img)
    ncond = NeuronConductance(m, m.linear)
    attribution = ncond.attribute(X_img, neuron_selector=1, target=1)
    attribution_np = np.transpose(
        attribution.squeeze().cpu().numpy(), axes=(1, 2, 0)
    )
    f = __plot(attribution_np)
    st.pyplot(f)  # very nice this plots the plt figure !
    st.write("Evaluation finished")        

# demo this only will work for saliency
def evaluate_button_saliency(
    input_image_path: str,
    model_path: str,
    loader_class_name: str,
    model_loader_path: str,
):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """
    m = __load_model(model_path, loader_class_name, model_loader_path)
    if m is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    saliency = Saliency(m.model)
    img = Image.open(input_image_path)
    img = np.array(img)  # convert to numpy array
    X_img = m.preprocess_image(img)
    attribution = saliency.attribute(X_img, target=0)
    attribution_np = np.transpose(
        attribution.squeeze().cpu().numpy(), axes=(1, 2, 0)
    )
    # the original image should have the (H,W,C) format
    f = __plot(attribution_np)
    st.pyplot(f)  # very nice this plots the plt figure !
    st.write("Evaluation finished")

def device_selection():
    # TODO
    options = ["CPU", "GPU(CUDA)"]
    selected_devices = st.sidebar.radio("choose a device:", options)
    if selected_devices == "CPU":
        # "Hier sollen über CPU implementiert werden"
        st.sidebar.write("you choose CPU")
    elif selected_devices == "GPU(CUDA)":
        # "Hier sollen über GPU implementiert werden"
        st.sidebar.write("you choose GPU(CUDA)")


def instances_selection():
    options = ["All", "Correct", "Incorrect"]
    selected_instances = st.sidebar.selectbox("Instances:", options)
    # Hier sollen über options für Instances implementiert werden
    if selected_instances == "All":
        st.sidebar.write("you choose All")
    elif selected_instances == "Correct":
        st.sidebar.write("you choose Correct")
    elif selected_instances == "Incorrect":
        st.sidebar.write("you choose Incorrect")


def upload_file(
    title: str, save_path: str, accept_multiple_files=False
) -> Optional[str | None]:
    """
    This method asks for a file and saves it to the specified path.

    Args:
        save_path (str): file path to save the uploaded file to.
    """
    global file_cache
    uploaded_file = st.file_uploader(title, accept_multiple_files=accept_multiple_files)
    if uploaded_file is not None:
        if save_path in file_cache:
            st.success("File loaded from cache")
            return file_cache[save_path]
        else:
            full_path = os.path.join(save_path, uploaded_file.name)
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            with open(full_path, "wb") as file:
                file.write(bytes_data)
            st.success("File saved successfully")

            file_cache[save_path] = full_path
            return full_path
    else:
        st.warning("No file uploaded")
        return None


def main():
    # Layout of the sidebar
    st.sidebar.title("Captum GUI")
    device_selection()
    st.sidebar.subheader("Filter by Instances")
    instances_selection()
    st.sidebar.subheader("Attribution Method Arguments")
    parameter_selection()
    # upload an image to test
    image_path = upload_file(
        "Upload an image",
        os.path.join(".", "captumcv", "image_tmp"),
        accept_multiple_files=False,
    )
    # upload function for the model
    model_path = upload_file(
        "Upload a model",
        os.path.join(".", "captumcv", "model_weights"),
        accept_multiple_files=False,
    )
    print(model_path)
    # upload model loader
    model_loader_path = upload_file(
        "Upload a model loader file",
        os.path.join(".", "captumcv", "loaders", "tmp"),
        accept_multiple_files=False,
    )
    # get all available classes from the model loader file
    print(model_loader_path)
    available_classes = []
    if model_loader_path is not None:
        available_classes = get_class_names_from_file(model_loader_path)
    # show class dropdown
    loader_class_name = st.selectbox("Select wanted class:", available_classes)
    st.write("You selected:", loader_class_name)
    col_eval = st.columns(1)[0]
    if col_eval.button("Evaluate"):
        print(choose_method)
        match choose_method:
            case Attr.SALIENCY.value:
                evaluate_button_saliency(
                    image_path, model_path, loader_class_name, model_loader_path
                )
            case Attr.IG.value:
                evaluate_button_ig(
                    image_path, model_path, loader_class_name, model_loader_path
                )
            case Attr.NEURON_CONDUCTANCE.value:
                evaluate_button_n_conductance(image_path, model_path, loader_class_name, model_loader_path)
            case Attr.NEURON_GUIDED_BACKPROPAGATION.value:
                pass
            case Attr.DECONVOLUTION.value:
                pass
            case Attr.TCAV.value:
                pass
            case Attr.GRADCAM.value:
                pass
            case _:
                st.write("No method selected")

if __name__ == "__main__":
    main()
