from tensorflow.keras import backend as K

def encode(tiles, model):
    features = model.predict(tiles)
    features = features.ravel()
    return features


class ResNet:
    __name__ = "ResNet"
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet, \
        decode_predictions

    def __init__(self, batch_size=1):
        self.model = ResNet50(include_top=False, weights='imagenet', pooling="avg")
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_resnet(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)

