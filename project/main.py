import os
import pickle
import uuid
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from captum.attr import (
    Deconvolution,
    GuidedGradCam,
    IntegratedGradients,
    NeuronConductance,
    NeuronGuidedBackprop,
    Saliency,
)
from captum.attr import visualization as viz
from PIL import Image

from captumcv.loaders.util.classLoader import (
    get_attribute_names_from_class,
    get_class_names_from_file,
    load_attribute_from_class,
    load_class_from_file,
)
from captumcv.loaders.util.modelLoader import ImageModelWrapper


class Attr(Enum):
    IG = "Integrated gradients"
    SALIENCY = "Saliency"
    TCAV_ALG = "TCAV"
    GRADCAM = "Guided GradCam"
    NEURON_CONDUCTANCE = "Neuron Conductance"
    NEURON_GUIDED_BACKPROPAGATION = "Neuron Guided Backpropagation"
    DECONVOLUTION = "Deconvolution"


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create directories for saving models and images
# BUG this is unexpected behaviour. os.path.join returning tuple. Steamlit might be involved! Need to report
CACHE_DIR = os.path.join(".cache")
CACHE_DIR = "".join(CACHE_DIR)
PATH_IMAGE_TMP = (os.path.join(".", "captumcv", "image_tmp"),)
PATH_IMAGE_TMP = "".join(PATH_IMAGE_TMP)
PATH_MODEL_WEIGHTS = (os.path.join(".", "captumcv", "model_weights"),)
PATH_MODEL_WEIGHTS = "".join(PATH_MODEL_WEIGHTS)
PATH_MODEL_LOADER = (os.path.join(".", "captumcv", "loaders", "tmp"),)
PATH_MODEL_LOADER = "".join(PATH_MODEL_LOADER)
PATH_MODEL_DEF_PY = (os.path.join(".", "captumcv", "models"),)
PATH_MODEL_DEF_PY = "".join(PATH_MODEL_DEF_PY)

os.makedirs(CACHE_DIR, exist_ok=True)

choose_method = st.selectbox(
    "Choose Attribution Method",
    (
        Attr.IG.value,
        Attr.SALIENCY.value,
        # Attr.TCAV_ALG.value,
        Attr.GRADCAM.value,
        Attr.NEURON_CONDUCTANCE.value,
        Attr.NEURON_GUIDED_BACKPROPAGATION.value,
        Attr.DECONVOLUTION.value,
    ),
)
st.write("You selected:", choose_method)
## Modell  und Parameter auswählen
# some variables for the parameter selection
# TODO This is very dangerous way of managing variables !! Just workaround for now
# TODO make this parameter recovery somehow smooth and nice.
nn_conductance_layer_options = []
choosen_layer = None
attr_dict = None
neuron_index = None
target_index = None

def __load_model(
    model_path: str, loader_class_name: str, model_loader_path: str
) -> Tuple[torch.nn.Module, ImageModelWrapper]:
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
            model = instance.model.module.to(
                "cpu"
            )  # if DataParallel do this to make it work on cpu
        else:
            model = instance.model
        return model, instance
    return None, None


def __plot(true_img, attr_img, flip_axis=True) -> "matplotlib.figure.Figure":
    # the original image should have the (H,W,C) format
    if flip_axis:
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


def __get_model_modules(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """
    This method returns all modules of the given model.
    Only load the attributes of instance torch.nn.Module.

    Args:
        model (torch.nn.Module): model to get the modules from

    Returns:
        Tuple[List[str],List[torch.nn.Module]]: returns a tuple of the module names and the modules
    """
    attr = get_attribute_names_from_class(model)
    nn_modules: List[
        torch.nn.Module
    ] = []  # contain objects that are of instance torch.nn.Module
    nn_modules_names: List[str] = []
    attr_name: str
    for attr_name in attr:
        attr_obj = load_attribute_from_class(model, attr_name)
        if isinstance(attr_obj, torch.nn.Module):
            nn_modules_names.append(attr_name)
            nn_modules.append(attr_obj)
    res_dict = dict(zip(nn_modules_names, nn_modules))
    return res_dict

# Funktion for Deconvolution
def evaluation_button_deconvolution(input_image_path: str,
    model_path: str,    loader_class_name: str,    model_loader_path: str,):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """
    model, model_loader = __load_model(model_path, loader_class_name, model_loader_path)
    if model is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    deconvolution = Deconvolution(model)
    img = Image.open(input_image_path)
    img = np.array(img)  # convert to numpy array
    X_img = model_loader.preprocess_image(image=img)
    attribution = deconvolution.attribute(X_img, target=0)
    attribution_np = np.transpose(
        attribution.squeeze().cpu().numpy(), axes=(1, 2, 0)
    )
    # the original image should have the (H,W,C) format
    f = __plot(img, attribution_np, flip_axis=False)
    st.pyplot(f)  # very nice this plots the plt figure !
    st.write("Evaluation finished")
    prediction = model(X_img)
    predicted_class = prediction.argmax(dim=1)
    st.markdown("<h2>Vorhersageklasse: {}</h2>".format(predicted_class.item()), unsafe_allow_html=True)


# Functin for Neuron BPB
def evaluate_button_guided_backprop(
    input_image_path: str, model_path: str, loader_class_name: str, model_loader_path,choosen_layer: str,neuron_index
):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """
    model, model_loader = __load_model(model_path, loader_class_name, model_loader_path)
    layer = load_attribute_from_class(model, choosen_layer)
    #layer = load_attribute_from_class(model)
    if model is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    img = Image.open(input_image_path)
    img = np.array(img)
    X_img = model_loader.preprocess_image(image=img)
    
    gbpp = NeuronGuidedBackprop(
        model,
        layer
    )
    neuron_index_cast = __try_convert_stt_to_int_or_tuple(neuron_index)
    if neuron_index_cast is None:
        st.warning("Failed to convert neuron index to int or tuple of ints")
    attribution = gbpp.attribute(X_img, neuron_selector=neuron_index_cast)
    attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), (1, 2, 0))
    f = __plot(img, attribution_np, flip_axis=False)
    st.pyplot(f)
    st.write("Evaluation finished")
    prediction = model(X_img)
    predicted_class = prediction.argmax(dim=1)
    st.markdown("<h2>Vorhersageklasse: {}</h2>".format(predicted_class.item()), unsafe_allow_html=True)

# Function for IG
def evaluate_button_ig(
    input_image_path: str, model_path: str, loader_class_name: str, model_loader_path: str, selected_method, selected_steps
):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """
    model, model_loader = __load_model(model_path, loader_class_name, model_loader_path)
    if model is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    ig = IntegratedGradients(model)
    img = Image.open(input_image_path)
    img = np.array(img)  # convert to numpy array
    X_img = model_loader.preprocess_image(image=img)
    #method_ig = parameter_selection()
    attribution = ig.attribute(X_img, target=0, method=selected_method,n_steps=selected_steps)
    st.write(selected_method)
    st.write(selected_steps)
    attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), (1, 2, 0))
    f = __plot(img, attribution_np)
    st.pyplot(f)
    st.write("Evaluation finished")
    prediction = model(X_img)
    predicted_class = prediction.argmax(dim=1)
    st.markdown("<h2>Vorhersageklasse: {}</h2>".format(predicted_class.item()), unsafe_allow_html=True)

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
    model, model_loader = __load_model(model_path, loader_class_name, model_loader_path)
    if model is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    saliency = Saliency(model)
    img = Image.open(input_image_path)
    img = np.array(img)  # convert to numpy array
    X_img = model_loader.preprocess_image(image=img)
    attribution = saliency.attribute(X_img, target=0)
    attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), axes=(1, 2, 0))
    # the original image should have the (H,W,C) format
    f = __plot(img, attribution_np)
    st.pyplot(f)  # very nice this plots the plt figure !
    st.write("Evaluation finished")
    prediction = model(X_img)
    predicted_class = prediction.argmax(dim=1)
    st.markdown("<h2>Vorhersageklasse: {}</h2>".format(predicted_class.item()), unsafe_allow_html=True)


def __convert_str_to_tuple(str_input: str) -> Tuple[int]:
    """
    Converts a string representation of a tuple of integers into an actual tuple.

    Args:
        str_input (str): The input string to convert.

    Returns:
        Tuple[int]: The converted tuple of integers.

    Raises:
        None.

    Examples:
        >>> __convert_str_to_tuple("(1, 2, 3)")
        (1, 2, 3)

        >>> __convert_str_to_tuple("[4, 5, 6]")
        (4, 5, 6)

        >>> __convert_str_to_tuple("{7, 8, 9}")
        (7, 8, 9)

    Notes:
        - The input string should have the format of a tuple containing integers.
        - The function removes any parentheses, brackets, braces, commas, semicolons, colons, and periods from the input string.
        - Each integer value in the string is separated by whitespace.
        - If the input string cannot be converted to a tuple of integers, None is returned.
    """
    try:
        str_input = str_input.replace("(", "")
        str_input = str_input.replace(")", "")
        str_input = str_input.replace("[", "")
        str_input = str_input.replace("]", "")
        str_input = str_input.replace("{", "")
        str_input = str_input.replace("}", "")
        str_input = str_input.replace(",", " ")
        str_input = str_input.replace(";", " ")
        str_input = str_input.replace(":", " ")
        str_input = str_input.replace(".", " ")
        return tuple(map(int, str_input.split(" ")))
    except ValueError:
        return None


def __convert_str_to_int(str_input: str) -> Optional[int | Tuple[int]]:
    """
    This method converts a string to an int or a tuple of ints.

    Args:
        str_input (str): string to convert

    Returns:
        Optional[int|Tuple[int]]: returns the converted string or None if the conversion failed
    """
    return int(str_input)


def __try_convert_stt_to_int_or_tuple(str_input: str) -> Optional[int | Tuple[int]]:
    """
    This method converts a string to an int or a tuple of ints.

    Args:
        str_input (str): string to convert
    Returns:
        Optional[int|Tuple[int]]: returns the converted string or None if the conversion failed
    """
    try:
        return __convert_str_to_int(str_input)
    except ValueError:
        return __convert_str_to_tuple(str_input)


def evaluate_button_neuron_conductance(
    input_image_path: str,
    model_path: str,
    loader_class_name: str,
    model_loader_path: str,
    choosen_layer: str,
    neuron_index: str,
    target_index: str,
):
    """
    This method runs the captum algorithm and shows the results.
    Use with DataParallel https://captum.ai/tutorials/Distributed_Attribution

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): chosen class loader name
        model_loader_path (str): model loader python file path
    """
    model, model_loader = __load_model(model_path, loader_class_name, model_loader_path)
    layer = load_attribute_from_class(model, choosen_layer)
    if model is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    img = Image.open(input_image_path)
    img = np.array(img)  # convert to numpy array
    X_img = model_loader.preprocess_image(image=img)
    ncond = NeuronConductance(model, layer)
    neuron_index_cast = __try_convert_stt_to_int_or_tuple(neuron_index)
    if neuron_index_cast is None:
        st.warning("Failed to convert neuron index to int or tuple of ints")
    if target_index is None:
        st.warning("Failed to convert target index to int or tuple of ints")
        return
    target_index_cast = __try_convert_stt_to_int_or_tuple(target_index)
    attribution = ncond.attribute(
        X_img, neuron_selector=neuron_index_cast, target=target_index_cast
    )
    attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), axes=(1, 2, 0))
    f = __plot(img, attribution_np, flip_axis=False)
    st.pyplot(f)  # very nice this plots the plt figure !
    st.write("Evaluation finished")
    prediction = model(X_img)
    predicted_class = prediction.argmax(dim=1)
    st.markdown("<h2>Vorhersageklasse: {}</h2>".format(predicted_class.item()), unsafe_allow_html=True)

def evaluate_button_gradcam(
    input_image_path: str,
    model_path: str,
    loader_class_name: str,
    model_loader_path: str,
    choosen_layer: str,
    target_index: str,
):
    """
    This method runs the captum algorithm and shows the results.
    Use with DataParallel https://captum.ai/tutorials/Distributed_Attribution

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): chosen class loader name
        model_loader_path (str): model loader python file path
    """
    model, model_loader = __load_model(model_path, loader_class_name, model_loader_path)
    layer = load_attribute_from_class(model, choosen_layer)
    if model is None:
        st.warning("Failed to load the class from the file. Try loading the file again")
        return
    img = Image.open(input_image_path)
    img = np.array(img)  # convert to numpy array
    X_img = model_loader.preprocess_image(image=img)
    guided_gc = GuidedGradCam(model, layer)

    if target_index is None:
        st.warning("Failed to convert target index to int or tuple of ints")
        return
    target_index_cast = __try_convert_stt_to_int_or_tuple(target_index)
    attribution = guided_gc.attribute(X_img, target_index_cast)
    attribution_np = np.transpose(
        attribution.squeeze().detach().cpu().numpy(), axes=(1, 2, 0)
    )
    #attribution_np = np.transpose(attribution.squeeze().cpu().numpy())
    f = __plot(img, attribution_np, flip_axis=False)
    st.pyplot(f)  # very nice this plots the plt figure !
    st.write("Evaluation finished")
    prediction = model(X_img)
    predicted_class = prediction.argmax(dim=1)
    st.markdown("<h2>Vorhersageklasse: {}</h2>".format(predicted_class.item()), unsafe_allow_html=True)

def device_selection():
    """
    Allows the user to select the device (CPU or GPU) for execution.

    Returns:
        str: Selected device ("cpu" for CPU, "cuda" for GPU).
    """
    options = ["CPU", "GPU(CUDA)"]
    selected_devices = st.sidebar.radio("choose a device:", options)
    if selected_devices == "CPU":
        # "Hier sollen über CPU implementiert werden"
        st.sidebar.write("you choose CPU")
        return "cpu"
    elif selected_devices == "GPU(CUDA)":
        # "Hier sollen über GPU implementiert werden"
        st.sidebar.write("you choose GPU(CUDA)")
        return "cuda"


def delete_cache():
    """
    Deletes cache files when the "Delete cache" button is clicked.

    Returns:
        None
    """
    if st.sidebar.button("Delete cache"):
        delete_files_except_gitkeep(CACHE_DIR)
        delete_files_except_gitkeep(PATH_MODEL_LOADER)
        delete_files_except_gitkeep(PATH_IMAGE_TMP)
        delete_files_except_gitkeep(PATH_MODEL_WEIGHTS)
        delete_files_except_gitkeep(PATH_MODEL_DEF_PY)
        st.sidebar.text("Cache deleted")


def delete_files_except_gitkeep(directory):
    """
    Deletes files in a directory, except for the ".gitkeep" file.

    Args:
        directory (str): Path to the directory.

    Returns:
        None
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file != ".gitkeep":
                file_path = os.path.join(root, file)
                os.remove(file_path)

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            delete_files_except_gitkeep(dir_path)


def upload_file(
    title: str, save_path: str, accept_multiple_files=False
) -> Optional[str | None]:
    """
    This method asks for a file and saves it to the specified path.

    Args:
        save_path (str): file path to save the uploaded file to.
    """
    cache_file_path = os.path.join(CACHE_DIR, f"{title}.pkl")
    uploaded_file = st.file_uploader(title, accept_multiple_files=accept_multiple_files)
    if uploaded_file is not None:
        delete_files_except_gitkeep(save_path)
        full_path = os.path.join(save_path, uploaded_file.name)
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        with open(full_path, "wb") as file:
            file.write(bytes_data)
            file.close()

        st.success("File saved successfully")
        # Save data to cache
        with open(cache_file_path, "wb") as cache_file:
            pickle.dump(full_path, cache_file)
        st.info("Saved to cache.")
        return full_path
    elif os.path.exists(cache_file_path):
        # Load data from cache
        with open(cache_file_path, "rb") as cache_file:
            cached_data = pickle.load(cache_file)
        st.info("Loaded from cache.")
        return cached_data
    else:
        st.warning("No file uploaded")
        delete_files_except_gitkeep(save_path)
        return None


def main():
    with st.sidebar.container():
        image = Image.open("D:\\Desktop\\group-1\\project\\build\\captum_logo.png")
        image = image.resize((190, 50)) 
        st.image(image, use_column_width=True)
    # Layout of the sidebar
    #st.sidebar.image("D:\\Desktop\\group-1\\project\\build\\captum_logo.png",use_column_width=True)
        #st.sidebar.markdown("<h1 style='font-size: 32px;'>Captum GUI</h1>", unsafe_allow_html=True)
    
    # device = device_selection()  # TODO this still need to be done
    delete_cache()
    
    # upload an image to test
    image_path = upload_file(
        "Upload an image",
        PATH_IMAGE_TMP,
        accept_multiple_files=False,
    )
    # upload function for the model
    model_path = upload_file(
        "Upload a model(.pth)",
        PATH_MODEL_WEIGHTS,
        accept_multiple_files=False,
    )
    # upload function for the model
    model_source_path = upload_file(
        "Upload a model def(.py)",
        PATH_MODEL_DEF_PY,
        accept_multiple_files=False,
    )
    # upload model loader
    model_loader_path = upload_file(
        "Upload a model loader file(.py)",
        PATH_MODEL_LOADER,
        accept_multiple_files=False,
    )
    # get all available classes from the model loader file
    available_classes = []
    if model_loader_path is not None:
        available_classes = get_class_names_from_file(model_loader_path)
    # show class dropdown
    loader_class_name = st.selectbox("Select wanted class:", available_classes)
    st.write("You selected:", loader_class_name)
    
    if choose_method == Attr.IG.value:
        st.sidebar.subheader("Attribution Method Arguments")
        method_options = ["gausslegendre", "riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid"]
        selected_method = st.sidebar.selectbox("Integrationsmethode", method_options)
        selected_steps = st.sidebar.number_input("Anzahl der Schritte", min_value=1, step=1)
    if choose_method == Attr.NEURON_GUIDED_BACKPROPAGATION.value:
        neuron_index = st.sidebar.text_input(
            "Insert neuron index (int, tuple[int]):", value="1")
        if model_loader_path is None:
            st.write("Please upload a model loader file first")
        else:
            st.warning("Loading the model to get the layers. This might take a while")
            model, _ = __load_model(model_path, loader_class_name, model_loader_path)
            if model is None:
                st.warning(
                    "Failed to load the class from the file. Try loading the file again"
                )
                return
            attr_dict = __get_model_modules(model)
            choosen_layer = st.sidebar.selectbox("Choose layer:", attr_dict.keys())
            st.sidebar.write(choosen_layer)
    if choose_method == Attr.NEURON_CONDUCTANCE.value:
        neuron_index = st.sidebar.text_input(
            "Insert neuron index (int, tuple[int]):", value="1"
        )
        target_index = st.sidebar.text_input(
            "Insert target index (int, tuple[int]):", value="1"
        )
        # update the list of layers in the options
        if model_loader_path is None:
            st.write("Please upload a model loader file first")
        else:
            st.warning("Loading the model to get the layers. This might take a while")
            model, _ = __load_model(model_path, loader_class_name, model_loader_path)
            if model is None:
                st.warning(
                    "Failed to load the class from the file. Try loading the file again"
                )
                return
            attr_dict = __get_model_modules(model)
            choosen_layer = st.sidebar.selectbox("Choose layer:", attr_dict.keys())
            st.sidebar.write(choosen_layer)
    if choose_method == Attr.GRADCAM.value:
        target_index = st.sidebar.text_input(
            "Insert target index (int, tuple[int]):", value="1"
        )
        # update the list of layers in the options
        if model_loader_path is None:
            st.write("Please upload a model loader file first")
        else:
            st.warning("Loading the model to get the layers. This might take a while")
            model, _ = __load_model(model_path, loader_class_name, model_loader_path)
            if model is None:
                st.warning(
                    "Failed to load the class from the file. Try loading the file again"
                )
                return
            attr_dict = __get_model_modules(model)
            choosen_layer = st.sidebar.selectbox("Choose layer:", attr_dict.keys())
            st.sidebar.write(choosen_layer)
    col_eval = st.columns(1)[0]
    if col_eval.button("Evaluate"):
        match choose_method:
            case Attr.SALIENCY.value:
                evaluate_button_saliency(
                    image_path, model_path, loader_class_name, model_loader_path
                )
            case Attr.IG.value:
                evaluate_button_ig(
                    image_path, model_path, loader_class_name, model_loader_path,selected_method, selected_steps
                )
            case Attr.NEURON_CONDUCTANCE.value:
                evaluate_button_neuron_conductance(
                    image_path,
                    model_path,
                    loader_class_name,
                    model_loader_path,
                    choosen_layer,
                    neuron_index,
                    target_index,
                )
            case Attr.NEURON_GUIDED_BACKPROPAGATION.value:
                evaluate_button_guided_backprop(
            image_path, model_path, loader_class_name, model_loader_path,choosen_layer,neuron_index
        )

            case Attr.DECONVOLUTION.value:
                evaluation_button_deconvolution(
                    image_path, model_path, loader_class_name, model_loader_path
                )  
            case Attr.TCAV_ALG.value:
                pass
            case Attr.GRADCAM.value:
                evaluate_button_gradcam(
                    image_path,
                    model_path,
                    loader_class_name,
                    model_loader_path,
                    choosen_layer,
                    target_index,
                )
            case _:
                st.write("No method selected")
    st.sidebar.markdown("""
    <h2>Predict List:</h2>
    <ul>
        <li>airplane: 0</li>
        <li>automobile: 1</li>
        <li>bird: 2</li>
        <li>cat: 3</li>
        <li>deer: 4</li>
        <li>dog: 5</li>
        <li>frog: 6</li>
        <li>horse: 7</li>
        <li>ship: 8</li>
        <li>truck: 9</li>
    </ul>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
