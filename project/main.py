import os
from typing import Optional, Tuple
import joblib
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from captum.attr import (
    GuidedBackprop,
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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
choose_method = st.selectbox(
    'Choose Attribution Method',
    ('Integrated gradients', 'Saliency', 'TCAV', 'GradCam', 'Neuron Conductance', 'Neuron Guided Backpropagation', 'Deconvolution'))
st.write('You selected:', choose_method)


file_cache = {}
## Modell  und Parameter auswählen 
def parameter_selection():
    if choose_method == "Integrated gradients":
        options = ["Gausslegendre", "Riemann_left",
                   "Riemann_right", "Riemann_middle", "Riemann_trapezoid"]
        st.sidebar.selectbox('method:', options)
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
        st.sidebar.number_input('Insert step:', min_value=25, step=1)
    if choose_method == "Seliency":
        st.sidebar.text("without parameter")
    if choose_method == "TCAV":
        #need parameter from TCAV
        st.sidebar.write("you choose TCAV") 
    if choose_method == "GradCam":
        #need parameter from GradCam
        st.sidebar.write("you choose GradCam")
    if choose_method =="Neuron Conductance":
        #need parameter from Neuron Conductance
        st.sidebar.write("you choose Neuron Conductance")
    if choose_method == "Neuron Guided Backpropagation":
        st.sidebar.text("without parameter")
    if choose_method == "Deconvolution":
        st.sidebar.text("without parameter")


def model_loader_class_button(uploaded_file):
    #global file_cache
    # hochladen oder angegebene Path
    path = 'project/testbild.jpg'
    image = Image.open(path)
    st.image(image, caption='origin Bild')

    if uploaded_file is not None:
        #model = load_model(uploaded_file)
        st.write("Image uploaded successfully")
    else:
        st.warning("No file uploaded")


def model_loaded_button(uploaded_file):
    # hochladen oder angegebene Path
    path = 'project/testbild.jpg'
    image = Image.open(path)
    st.image(image, caption='origin Bild')

    if uploaded_file is not None:
        st.write("Image uploaded successfully")
    else:
        st.warning("No file uploaded")


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
            transforms.Normalize(mean = [ 0., 0., 0. ],
                                std = [ 1/0.2023, 1/0.1994, 1/0.2010]),
        transforms.Normalize(mean = [-0.4914, -0.4822, -0.4465],
                            std = [ 1., 1., 1. ]),
        ]
    )

    x_img_inv = inv_normal(x_img_before)

    return x_img, x_img_before, x_img_inv
#Funktion for Deconvolution
def evaluation_button_deconvolution(input_img_path:str, model_path:str, loader_class_name:str,model_loader_path):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): chosen class loader name
        model_loader_path (str): model loader python file path
    """
    model_loader = load_class_from_file(model_loader_path, loader_class_name)
    if model_loader and issubclass(model_loader,ImageModelWrapper):
        instance: ImageModelWrapper = model_loader(model_path)
        tmp_model = instance.model
        deconvolution = Deconvolution(instance.model)
        x_img, x_img_before, x_img_inv = process_image(input_img_path, instance.get_image_shape())
        attribution = deconvolution.attribute(x_img, target = 0)
        attribution_np = np.transpose(attribution.squeeze().cpu().numpy(),(1,2,0))
        print(attribution.shape)
        print(attribution_np.shape)
        f,ax = viz.visualize_image_attr_multiple(attribution_np,
            x_img_inv.permute(1,2,0).numpy(), 
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar = True,
            outlier_perc = 2, 
        )
        st.pyplot(f) 
        st.write("Evaluation finished")
    else:
        st.warning(
            "Failed to load the class from the file. Try loading the file again")



#Funktion for Neuron Guided Backpropagation
def evaluate_button_ngbp(input_img_path:str, model_path:str, loader_class_name:str,model_loader_path):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): chosen class loader name
        model_loader_path (str): model loader python file path
    """
    model_loader = load_class_from_file(model_loader_path, loader_class_name)
    if model_loader and issubclass(model_loader,ImageModelWrapper):
        instance: ImageModelWrapper = model_loader(model_path)
        tmp_model = instance.model
        guided_BP = GuidedBackprop(instance.model)
        x_img, x_img_before, x_img_inv = process_image(input_img_path, instance.get_image_shape())
        attribution = guided_BP.attribute(x_img, target = 0)
        attribution_np = np.transpose(attribution.squeeze().cpu().numpy(),(1,2,0))
        print(attribution.shape)
        print(attribution_np.shape)
        f,ax = viz.visualize_image_attr_multiple(attribution_np,
            x_img_inv.permute(1,2,0).numpy(), 
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar = True,
            outlier_perc = 2, 
        )
        st.pyplot(f) 
        st.write("Evaluation finished")
    else:
        st.warning(
            "Failed to load the class from the file. Try loading the file again")
#Function for IG
def evaluate_button_ig(input_img_path:str, model_path:str, loader_class_name:str,model_loader_path):
    """
    This method runs the captum algorithm and shows the results.

    Args:
        model_path (str): Path to the model weights
        loader_class_name (str): choosen class loader name
        model_loader_path (str): model loader python file path
    """
    
    model_loader = load_class_from_file(model_loader_path, loader_class_name)
    if model_loader and issubclass(model_loader, ImageModelWrapper):
        instance:ImageModelWrapper = model_loader(model_path)
        tmp_model = instance.model
        ig = IntegratedGradients(instance.model)
        x_img, x_img_before,x_img_inv = process_image(input_img_path, instance.get_image_shape())
        attribution = ig.attribute(x_img, target = 0)
        attribution_np = np.transpose(attribution.squeeze().cpu().numpy(),(1,2,0))
        print(attribution.shape)
        print(attribution_np.shape)
        f,ax = viz.visualize_image_attr_multiple(attribution_np,
                                      x_img_inv.permute(1, 2, 0).numpy(),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )
        
        st.pyplot(f) 
        st.write("Evaluation finished")
    else:
        st.warning(
            "Failed to load the class from the file. Try loading the file again")
# demo this only will work for saliency
def evaluate_button_saliency(input_image_path: str, model_path: str, loader_class_name: str, model_loader_path: str):
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
        x_img, x_img_before, x_img_inv = process_image(input_image_path, instance.get_image_shape())
        attribution = saliency.attribute(x_img, target=0)
        attribution_np = np.transpose(attribution.squeeze().cpu().numpy(), (1,2,0))
        # print(attribution)
        print(attribution.shape)
        print(attribution_np.shape) # this does work
        f, ax = viz.visualize_image_attr_multiple(attribution_np,
                                      x_img_inv.permute(1, 2, 0).numpy(),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )
        
        st.pyplot(f) # very nice this plots the plt figure !

        # st.image(attribution_np,caption='origin Bild', width=300)
        # Now you can work with the dynamically loaded class instance
        st.write("Evaluation finished")
    else:
        st.warning(
            "Failed to load the class from the file. Try loading the file again")


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


def upload_file(title: str, save_path: str, accept_multiple_files=False) -> Optional[str | None]:
    """
    This method asks for a file and saves it to the specified path.

    Args:
        save_path (str): file path to save the uploaded file to.
    """
    global file_cache
    uploaded_file = st.file_uploader(
        title, accept_multiple_files=accept_multiple_files)
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
        "Upload an image", os.path.join(".","captumcv","image_tmp"), accept_multiple_files=False)

    # upload function for the model
    model_path = upload_file(
        "Upload a model", os.path.join(".","captumcv","model_weights"), accept_multiple_files=False)
    print(model_path)
    # upload model loader
    model_loader_path = upload_file(
        "Upload a model loader file", os.path.join(".","captumcv","loaders", "tmp"), accept_multiple_files=False)
    # get all available classes from the model loader file
    print(model_loader_path)
    available_classes = []
    if model_loader_path is not None:
        available_classes = get_class_names_from_file(model_loader_path)
    # show class dropdown
    loader_class_name = st.selectbox('Select wanted class:', available_classes)
    st.write('You selected:', loader_class_name)
    col_eval = st.columns(1)[0]
    if choose_method =='Saliency':
        if col_eval.button("Evaluate"):
            evaluate_button_saliency(image_path, model_path, loader_class_name, model_loader_path)
    elif choose_method =='Integrated gradients':
        if col_eval.button("Evaluate"):
            evaluate_button_ig(image_path, model_path, loader_class_name, model_loader_path)
    elif choose_method =='Neuron Guided Backpropagation':
        if col_eval.button("Evaluate"):
            evaluate_button_ngbp(image_path, model_path, loader_class_name, model_loader_path)
    elif choose_method =='Deconvolution':
        if col_eval.button("Evaluate"):
            evaluation_button_deconvolution(image_path, model_path, loader_class_name, model_loader_path)

if __name__ == "__main__":
    main()
