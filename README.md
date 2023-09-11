# Captumcv frontend for captum for working with computer vision models

This project aims to provide an intuitive way of using different Explainable AI methods to analyst computer vision models.

## Requirements

The code was tested with python 3.10 and ubuntu 20.04

## Installation

1. clone the repository

```
git clone <repository-url>
```

2. Install requirements by running:

```
python setup.py install
```

## Usage

To start the frontend run the following command:

```
streamlit run main.py
```

After that the consol will show the url to the frontend.
e.g http://localhost:8501

To attribute a model you need to upload:

1. Image file
2. Model file
3. The model definition file
4. Custom model loader file

After that you can select the method you want to use and the parameters for the method.
Currently supported attribution methods are:

1. Integrated Gradients
2. Saliency
3. TCAV [todo]
4. GradCam
5. Neuron Conductance
6. Neuron Guided Backpropagation
7. Deconvolution

For more documentation visit the [captum library](https://captum.ai/)

### Prediction

To predict the output of the model you can use the predict checkbox.
Currently only classification models are supported.
If you would like see the labels of the model you can upload a json file with the labels.
Fallowing should be the layout of the json file:

```
{
    "num_classes": 10,
    "classes": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        ...
    ]
}
```

### Writing custom loader

To write a custom loader you need to create a class that inherits from the ImageModelWrapper class. Because the python file is loaded dynamically the class needs to have import and inheritance predifined.

For such example see "resources/ResNet18Loader.py"

In the `__init__` method the input shape, target image shape, and normalization parameters need to be provided. For custom normalization the `preprocess_image` method can to be overwritten.

The resources contains two examples.
The corresponding models can be downloaded from [here](https://dlcv2023.s3.eu-north-1.amazonaws.com/model_weights.zip) (while link is still available)

# Contributing

To contribute to this project please open an issue and if possible a pull request.

# License

MIT
