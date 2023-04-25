
import streamlit as st
from PIL import Image
from fastai.vision.all import PILImage, load_learner, untar_data, get_image_files, ImageDataLoaders, ClassificationInterpretation, Resize, cnn_learner, URLs, resnet18, error_rate
import random
import os
import pathlib 
import torch

path = './oxford-iiit-pet/images'
imgs = get_image_files(path)
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(path, imgs, valid_pct=0.2, 
    seed=42,label_func=is_cat, item_tfms=Resize(224))
idx = random.randint(0, len(imgs))


#image = Image.open('D:/Desktop/SS2023/Deep Learning in VC/oxford-iiit-pet/images/Abyssinian_221.jpg')
image = PILImage.create(imgs[idx])

st.image(image, caption='original Bild')
#learn_inf = load_learner(cpu=False, fname='D:/Desktop/SS2023/Deep Learning in VC/resnet18_finetuned.pkl')

pathlib.PosixPath = pathlib.WindowsPath
learn_inf = load_learner('D:/Desktop/SS2023/Deep Learning in VC/resnet18_finetuned.pkl')

st.write("load model")
image = learn_inf.dls.after_item(image)
image = learn_inf.dls.after_batch(image.cuda())
pred,pred_idx,probs = learn_inf.predict(imgs[idx])
st.write(pred, pred_idx, probs)