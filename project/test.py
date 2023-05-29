import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients
model = torch.load('D:/Desktop/group-1/project/captumcv/model_weights/SimpleDLA_10epochs_cifar10.pth')
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)
ig = IntegratedGradients(model)

def compute_attributions(image):
    transformed_image = transform_image(image)
    attributions, _ = ig.attribute(transformed_image, target=0, return_convergence_delta=True)
    return attributions.squeeze().detach().numpy()
def main():
    st.title("Integrated Gradients")

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image")

        attributions = compute_attributions(image)
        st.image(attributions, caption="Attributions")
    else:
        st.warning("Please upload an image")

if __name__ == "__main__":
    main()
