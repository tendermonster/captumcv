import ast
import importlib.util

from captumcv.loaders.util.modelLoader import ImageModelWrapper


def load_class_from_file(file_path, class_name):
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_obj = getattr(module, class_name, None)
    return class_obj


def get_class_names_from_file(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())
    class_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
    return class_names


if __name__ == "__main__":
    # Example usage
    # file path must be available to the script
    file_path = "./captumcv/loaders/DLASimpleLoader.py"
    class_name = "DLASimpleLoader"  # the file name should be the same as the class name
    # path to the model weights
    path = "./captumcv/model_weights/SimpleDLA_10epochs_cifar10.pth"

    loaded_class = load_class_from_file(file_path, class_name)
    if loaded_class and isinstance(loaded_class, ImageModelWrapper):
        instance = loaded_class(path)
        print(instance.image_shape)
        # Now you can work with the dynamically loaded class instance
    else:
        print("Failed to load the class from the file.")
