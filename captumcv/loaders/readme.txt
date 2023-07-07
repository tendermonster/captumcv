loaders/ # contains  # prefferebly loaders should end with _loader.py to be automatically identified
    # use this https://docs.python.org/3/library/abc.html

    - abstractModelLoader.py # is an abstract class that requires you to
    # implement the get_model() -> torch model and get_input(img: ndarray) -> input

    - modelXYZ_loader.py # extends from abstractModelLoader