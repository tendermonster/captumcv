import streamlit as st

# Set page title

st.title("Bild-Upload")

# Create a file uploader widget
image = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

# Display the selected image
if image is not None:
    st.image(image, caption="Das hochgeladene Bild", use_column_width=True)
