from captumcv.loaders.modelLoader import DLASimpleLoader
from captumcv.loaders.modelLoader import ImageModelWrapper
import torch
import torchvision.transforms as transforms
import numpy as np
import streamlit as st
import os
from PIL import Image
from captum.attr import IntegratedGradients

#Attribution Method auswählen
choose_method = st.selectbox(
    'Choose Attribution Method',
    ('Integrated gradients', 'Seliency', 'TCAV', 'GradCam','Neuron Conductance','Neuron Guided Backpropagation','Deconvolution'))
st.write('You selected:', choose_method)


## Modell  und Parameter auswählen 
def parameter_selection():
    if choose_method == "Integrated gradients":
        options = ["Gausslegendre","Riemann_left","Riemann_right","Riemann_middle","Riemann_trapezoid"]
        st.sidebar.selectbox('method:',options)
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
        st.sidebar.number_input('Insert step:',min_value=25, step = 1)
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
    if choose_method =="Neuron Guided Backpropagation":
        st.sidebar.text("without parameter")
    if choose_method =="Deconvolution":
        st.sidebar.text("without parameter")

# choose_method = st.selectbox(
#     'Choose Attribution Method',
#     ('Integrated gradients', 'Seliency', 'TCAV', 'GradCam','Neuron Conductance','Neuron Guided Backpropagation','Deconvolution'))
# st.write('You selected:', choose_method)

# def transform_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#     ])
#     return transform(image).unsqueeze(0)
# #ig = IntegratedGradients(model)

# def compute_attributions(image):
#     transformed_image = transform_image(image)
#     attributions, _ = ig.attribute(transformed_image, target=0, return_convergence_delta=True)
#     return attributions.squeeze().detach().numpy()


def img_upload():
    uploaded_files = st.file_uploader("Choose a Picture", accept_multiple_files=True)
    images = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            st.write("Image uploaded successfully")
            images.append(uploaded_file)
            st.image(uploaded_file,caption='origin Bild')
        else:
            st.warning("No file uploaded")
    return images


def model_upload():
    uploaded_files = st.file_uploader("Choose Model(s)", accept_multiple_files=True)
    model_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            st.write("Model uploaded successfully")
            model_filename = uploaded_file.name
            model_path = "D:/Desktop/group-1/project/captumcv/model_weights/" + model_filename
            with open(model_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            model_paths.append(model_path)
        else:
            st.warning("No file uploaded")
    return model_paths


def compute_attributions(img_list):
    model = torch.load("D:/Desktop/group-1/project/captumcv/save_models/SimpleDLA_10epochs_cifar10.pth")
    for image_file in img_list:
        img = Image.open(image_file)
    model_path =", ".join(model)
    #model = torch.load(model_path)
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(img, target=0, return_convergence_delta=True)
    return attributions.squeeze().detach().numpy()

def evaluate_button(img_list,model):
    #ig = IntegratedGradients(model)
    for image_file in img_list:
        img = Image.open(image_file)
    model_path =", ".join(model)
    
    #wrapper = ImageModelWrapper(image_shape, model_path, model)

    model_loader = DLASimpleLoader(model_path)
    #model_loader = IntegratedGradients(model_loader)   
    #  
   
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),  # in case of cifar10
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]
    )
    
    x_img_test = transform_test(img)
    x_img_test = torch.reshape(x_img_test, model_loader.get_image_shape())
    # we may need to normalize the image here
    #st.write(x_img_test)
    image_shape = x_img_test.shape

    model_loader = ImageModelWrapper(image_shape, model_path, model)
 

    


    #st.write(x_img_test.shape)
    #y = image_model.predict()
    y = model_loader.predict(x_img_test)
    #objekt = ImageModelWrapper(x_img_test.shape,model_path,model_loader)
    classes = ("plane","car","bird","cat","deer",
        "dog","frog","horse","ship","truck",)

    st.subheader("Input Information")
    st.text(f"Shape of input image: {x_img_test.shape}")
    st.text(f"Classes: {classes}")
    st.text(f"Size of output tensor: {y.size()}")

    st.subheader("Prediction")
    predicted_class_index = y.argmax()
    predicted_class = classes[predicted_class_index]
    st.text(f"Predicted class index: {predicted_class_index}")
    st.text(f"Predicted class: {predicted_class}")
    st.write("Evaluation finished")

def device_selection():
    options = ["CPU", "GPU(CUDA)"]
    selected_devices = st.sidebar.radio("choose a device:",options)
    if selected_devices == "CPU":
        #"Hier sollen über CPU implementiert werden"
        st.sidebar.write("you choose CPU")
    elif selected_devices == "GPU(CUDA)":
        #"Hier sollen über GPU implementiert werden"
        st.sidebar.write("you choose GPU(CUDA)")


def instances_selection():
    options = ["All","Correct","Incorrect"]
    selected_instances = st.sidebar.selectbox("Instances:",options)
    # Hier sollen über options für Instances implementiert werden
    if selected_instances == "All":
        st.sidebar.write("you choose All")
    elif selected_instances == "Correct":
        st.sidebar.write("you choose Correct")
    elif selected_instances == "Incorrect":
        st.sidebar.write("you choose Incorrect")




def main():
    
    st.sidebar.title("Captum GUI")
    device_selection()
    st.sidebar.subheader("Filter by Instances")
    instances_selection()
    st.sidebar.subheader("Attribution Method Arguments")
    parameter_selection()
    model_loader = model_upload()
    uploaded_images = img_upload()
    col1, col2 = st.columns([1, 1])
    if col1.button("Evaluate"):
        evaluate_button(uploaded_images, model_loader)

if __name__ == "__main__":
    main()