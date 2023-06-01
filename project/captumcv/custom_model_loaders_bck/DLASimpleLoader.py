# example model definition
from captumcv.loaders.util.modelLoader import ImageModelWrapper
from captumcv.models.dla_simple import SimpleDLA


class DLASimpleLoader(ImageModelWrapper):
    def __init__(self, model_path):
        # TODO somehow automate the selection or import of SimpleDLA class ?
        model = SimpleDLA()
        image_shape = [1, 3, 32, 32]
        super(DLASimpleLoader, self).__init__(image_shape, model_path, model)
