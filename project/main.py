import streamlit as st
import os
from PIL import Image

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

def model_loaded_button(uploaded_file):
    # hochladen oder angegebene Path
    path = 'project/testbild.jpg'
    image = Image.open(path)
    st.image(image, caption='origin Bild')
    
    if uploaded_file is not None:
        st.write("Image uploaded successfully")
    else:
        st.warning("No file uploaded")
def evaluate_button():
    # hier implementieren das Prozess für Evaluation
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
    uploaded_file = st.file_uploader("Upload a model",accept_multiple_files=True)
    col1, col2 = st.columns(2)
    if col1.button("laden Bild"):
        model_loaded_button(uploaded_file)
    if col2.button("evaluate"):
        evaluate_button()

if __name__ == "__main__":
    main()