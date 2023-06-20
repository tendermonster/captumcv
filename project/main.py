import os
import pickle
import shutil
from typing import Optional, Tuple
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from captum.attr import (
    Deconvolution,
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
)
from captum.attr import visualization as viz
from PIL import Image
from captum.attr import IntegratedGradients
from captumcv.loaders.util.classLoader import (
    get_class_names_from_file,
    load_class_from_file,
)
from captumcv.loaders.util.modelLoader import ImageModelWrapper
from captum.attr import NeuronGuidedBackprop

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

os.makedirs(CACHE_DIR, exist_ok=True)

choose_method = st.selectbox(
    "Choose Attribution Method",
    (
        "Integrated gradients",  # Manyue
        "Seliency",  # Artiom
        "TCAV",  # Babar
        "GradCam",  # Babar
        "Neuron Conductance",  # Artiom
        "Neuron Guided Backpropagation",  # Manyue
        "Deconvolution",
    ),
)
st.write("You selected:", choose_method)


## Modell und Parameter auswählen
def parameter_selection():
    if choose_method == "Integrated gradients":
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
    if choose_method == "Seliency":
        st.sidebar.text("without parameter")
    if choose_method == "TCAV":
        # need parameter from TCAV
        st.sidebar.write("you choose TCAV")
    if choose_method == "GradCam":
        # need parameter from GradCam
        st.sidebar.write("you choose GradCam")
    if choose_method == "Neuron Conductance":
        # need parameter from Neuron Conductance
        st.sidebar.write("you choose Neuron Conductance")
    if choose_method == "Neuron Guided Backpropagation":
        st.sidebar.text("without parameter")


def process_image(image_path: str, image_shape: Tuple):
    """
    This method processes the image and returns the tensor of the correct shape.

    Args:
        image_path (str): path to image
        image_shape (Tuple): nn model input shape

    Returns:
        _type_: Tuple[x_img, x_img_before, x_img_inv]
    """
    img = Image.open(image_path)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),  # in case of cifar10
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    x_img_before = transform_test(img)
    # reshape to correct shape
    x_img = torch.reshape(x_img_before, image_shape)

    inv_normal = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
            ),
            transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465], std=[1.0, 1.0, 1.0]),
        ]
    )

    x_img_inv = inv_normal(x_img_before)

    return x_img, x_img_before, x_img_inv


# Funktion for Deconvolution
def evaluation_button_deconvolution(
    input_img_path: str, model_path: str, loader_class_name: str, model_loader_path
):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): chosen class loader name
        model_loader_path (str): model loader python file path
    """
    model_loader = load_class_from_file(model_loader_path, loader_class_name)
    if model_loader and issubclass(model_loader, ImageModelWrapper):
        instance: ImageModelWrapper = model_loader(model_path)
        tmp_model = instance.model
        deconvolution = Deconvolution(instance.model)
        x_img, x_img_before, x_img_inv = process_image(
            input_img_path, instance.get_image_shape()
        )
        attribution = deconvolution.attribute(x_img, target=0)
        attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), (1, 2, 0))
        print(attribution.shape)
        print(attribution_np.shape)
        f, ax = viz.visualize_image_attr_multiple(
            attribution_np,
            x_img_inv.permute(1, 2, 0).numpy(),
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar=True,
            outlier_perc=2,
        )
        st.pyplot(f)
        st.write("Evaluation finished")
    else:
        st.warning("Failed to load the class from the file. Try loading the file again")


# Functin for Neuron BPB
def evaluate_button_guided_backprop(
    input_img_path: str, model_path: str, loader_class_name: str, model_loader_path
):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """
    model_loader = load_class_from_file(model_loader_path, loader_class_name)
    if model_loader and issubclass(model_loader, ImageModelWrapper):
        instance: ImageModelWrapper = model_loader(model_path)
        tmp_model = instance.model
        model = tmp_model.module.to("cpu")
        gbpp = NeuronGuidedBackprop(
            instance.model,
            model.linear,
        )
        x_img, x_img_before, x_img_inv = process_image(
            input_img_path, instance.get_input_shape()
        )
        attribution = gbpp.attribute(x_img, neuron_selector=1)
        attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), (1, 2, 0))
        f, ax = viz.visualize_image_attr_multiple(
            attribution_np,
            x_img_inv.permute(1, 2, 0).numpy(),
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar=True,
            outlier_perc=2,
        )

        st.pyplot(f)
        st.write("Evaluation finished")
    else:
        st.warning("Failed to load the class from the file. Try loading the file again")


# Function for IG
def evaluate_button_ig(
    input_img_path: str, model_path: str, loader_class_name: str, model_loader_path
):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """

    model_loader = load_class_from_file(model_loader_path, loader_class_name)
    if model_loader and issubclass(model_loader, ImageModelWrapper):
        instance: ImageModelWrapper = model_loader(model_path)
        tmp_model = instance.model
        ig = IntegratedGradients(instance.model)
        x_img, x_img_before, x_img_inv = process_image(
            input_img_path, instance.get_input_shape()
        )
        attribution = ig.attribute(x_img, target=0)
        attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), (1, 2, 0))
        f, ax = viz.visualize_image_attr_multiple(
            attribution_np,
            x_img_inv.permute(1, 2, 0).numpy(),
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar=True,
            outlier_perc=2,
        )

        st.pyplot(f)
        st.write("Evaluation finished")
    else:
        st.warning("Failed to load the class from the file. Try loading the file again")


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
    model_loader = load_class_from_file(model_loader_path, loader_class_name)
    # check that the class extends correct subclass
    if model_loader and issubclass(model_loader, ImageModelWrapper):
        instance: ImageModelWrapper = model_loader(model_path)
        tmp_model = instance.model
        saliency = Saliency(instance.model)
        # saliency = IntegratedGradients(instance.model)
        x_img, x_img_before, x_img_inv = process_image(
            input_image_path, instance.get_input_shape()
        )
        attribution = saliency.attribute(x_img, target=0)
        attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), (1, 2, 0))
        f, ax = viz.visualize_image_attr_multiple(
            attribution_np,
            x_img_inv.permute(1, 2, 0).numpy(),
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar=True,
            outlier_perc=2,
        )

        st.pyplot(f)  # very nice this plots the plt figure !

        # st.image(attribution_np,caption='origin Bild', width=300)
        # Now you can work with the dynamically loaded class instance
        st.write("Evaluation finished")
    else:
        st.warning("Failed to load the class from the file. Try loading the file again")


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


def delete_cache():
    if st.sidebar.button("Delete cache"):
        delete_files_except_gitkeep(CACHE_DIR)
        delete_files_except_gitkeep(PATH_MODEL_LOADER)
        delete_files_except_gitkeep(PATH_IMAGE_TMP)
        delete_files_except_gitkeep(PATH_MODEL_WEIGHTS)
        st.sidebar.text("Cache deleted")


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


def delete_files_except_gitkeep(directory):
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
    # Layout of the sidebar
    st.sidebar.title("Captum GUI")
    device_selection()
    delete_cache()
    st.sidebar.subheader("Filter by Instances")
    instances_selection()
    st.sidebar.subheader("Attribution Method Arguments")
    parameter_selection()
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
    col_eval = st.columns(1)[0]
    if choose_method == "Seliency" and col_eval.button("Evaluate"):
        evaluate_button_saliency(
            image_path, model_path, loader_class_name, model_loader_path
        )
    elif choose_method == "Integrated gradients" and col_eval.button("Evaluate"):
        evaluate_button_ig(image_path, model_path, loader_class_name, model_loader_path)
    elif choose_method == "Deconvolution" and col_eval.button("Evaluate"):
        evaluate_button_ig(image_path, model_path, loader_class_name, model_loader_path)
    elif choose_method == "Neuron Guided Backpropagation" and col_eval.button(
        "Evaluate"
    ):
        evaluate_button_guided_backprop(
            image_path, model_path, loader_class_name, model_loader_path
        )


if __name__ == "__main__":
    main()
