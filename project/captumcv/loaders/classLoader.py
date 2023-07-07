import ast
import importlib.util
import typing
from typing import Any, List

from captumcv.loaders.modelLoader import ImageModelWrapper


def load_class_from_file(file_path: str, class_name: str) -> Any:
    """
    Dynamically loads a class from a file.

    Args:
        file_path (str): Path to the file containing the class definition.
        class_name (str): Name of the class to load.

    Returns:
        Any: The loaded class object, or None if loading fails.
    """
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_obj = getattr(module, class_name, None)
    return class_obj


def load_attribute_from_class(class_obj: Any, attribute_name: str) -> Any:
    """
    Loads a specific attribute from a class.

    Args:
        class_obj (Any): The class object.
        attribute_name (str): Name of the attribute to load.

    Returns:
        Any: The loaded attribute, or None if the attribute doesn't exist.
    """
    return getattr(class_obj, attribute_name, None)


def get_attribute_names_from_class(class_obj: Any) -> List[str]:
    """
    Retrieves the names of all attributes within a class.

    Args:
        class_obj (Any): The class object.

    Returns:
        List[str]: A list of attribute names.
    """
    return dir(class_obj)


def get_class_names_from_file(file_path: str) -> List[str]:
    """
    Extracts the names of all classes defined in a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        List[str]: A list of class names.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())
    class_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
    return class_names
