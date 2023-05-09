import streamlit as st
import os
from PIL import Image

# Using "with" notation
with st.sidebar:
    st.title("Captum GUI")

    add_radio = st.radio(
        "choose a device:",
        ("CPU ", "GPU(CUDA)")
    )


option_a = st.selectbox(
    'Choose a algorithm',
    ('Integrated gradients', 'Seliency', 'TCAV ', 'GradCam','Neuron Conductance','Neuron Guided Backpropagation and Deconvolution'))

uploaded_files = st.file_uploader("Choose a Modell", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)
## Modell auswählen filechoose
values = st.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

st.write('You selected:', option_a)

if st.button('yes'):
    path = 'D:/Desktop/group-1/project/testbild.jpg'
    if not os.path.exists(path):
        print("give a user a notification that file path does not exist")
    # check if the file format is correct by checking if image is jpg or JPG or png or or or 
    image = Image.open('D:/Desktop/group-1/project/testbild.jpg')
    st.image(image, caption='origin Bild')
    st.write('Bilder werden verarbeitet')
else:
    st.write('try again')


try:
    path = 'D:/Desktop/group-1/project/testbild.jpg'
    image = Image.open(path)
except FileNotFoundError as e:
    print("image error")



# import streamlit as st
# import numpy as np
# import pandas as pd
# import torch
# import captum
# from captum.attr import IntegratedGradients
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Generate synthetic data
# X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a binary classification model
# clf = LogisticRegression(random_state=42)
# clf.fit(X_train, y_train)

# # Define the Streamlit app
# st.title("Captum Integrated Gradients Demo")
# st.sidebar.title("Demo Options")

# # Define the input widgets
# input_data = st.sidebar.text_area("Input Data", "1, 2, 3, 4, 5")
# input_data = np.fromstring(input_data, dtype=float, sep=',').reshape(1, -1)
# attribution_method = st.sidebar.selectbox("Attribution Method", ["Integrated Gradients"])
# baseline = st.sidebar.slider("Baseline", min_value=-5.0, max_value=5.0, step=0.1, value=0.0)
# num_steps = st.sidebar.slider("Number of Steps", min_value=1, max_value=100, step=1, value=50)

# # Define the output
# st.subheader("Model Output")
# model_output = clf.predict(input_data)
# st.write(model_output)

# # Define the attribution calculation
# if attribution_method == "Integrated Gradients":
#     ig = IntegratedGradients(clf.predict_proba)
#     attributions = ig.attribute(torch.tensor(input_data), torch.tensor([[baseline]*input_data.shape[1]]), 
#                                 n_steps=num_steps)
#     attributions = attributions.numpy().flatten()
#     attributions_df = pd.DataFrame({'Feature': range(1, input_data.shape[1]+1),
#                                     'Importance': attributions})
#     st.subheader("Attributions")
#     st.dataframe(attributions_df)

# # Define the accuracy calculation
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# st.sidebar.subheader("Model Performance")
# st.sidebar.write("Accuracy:", accuracy)

# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import captum
# from captum.attr import IntegratedGradients

# # Load the image
# image_path = "Download.jpeg"
# image = Image.open(image_path)

# # Preprocess the image
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# image_tensor = transform(image).unsqueeze(0)

# # Load the model
# map_location=torch.device('cpu') 
# model = torch.load("resnet18_finetuned.pkl")

# # Define the attribution method
# ig = IntegratedGradients(model)

# # Compute the attributions
# attributions = ig.attribute(image_tensor, target=1)

# # Visualize the attributions
# attributions = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
# plt.imshow(np.clip(attributions, 0, 1))
# plt.axis('off')
# plt.show()
# # 