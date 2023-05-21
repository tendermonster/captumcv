from captumcv.loaders.modelLoader import DLASimpleLoader
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

if __name__ == "__main__":
    model_path = os.path.join(
        "captumcv", "model_weights", "SimpleDLA_10epochs_cifar10.pth"
    )
    model_loader = DLASimpleLoader(model_path)

    # prepare image for input
    # img = Image.open("resources/airplain.jpg")
    # img = Image.open("resources/ship.jpg")
    img = Image.open(
        os.path.join("resources", "testbild.jpg")
    )  # Image with different size
    # the transforms.Compose should handle transformations
    # x_img = np.array(img)
    # x_img = np.reshape(x_img, model_loader.get_image_shape())
    # x_img = torch.from_numpy(x_img)
    # normalizes the image
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),  # in case of cifar10
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    x_img_test = transform_test(img)
    # reshape to correct shape
    x_img_test = torch.reshape(x_img_test, model_loader.get_image_shape())
    # we may need to normalize the image here
    print(x_img_test.shape)
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    y = model_loader.predict(x_img_test)
    print(y.size())
    print(y.argmax())
