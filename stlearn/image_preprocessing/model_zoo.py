class Model:
    __name__ = "CNN base model"

    def __init__(self, base, batch_size=1):
        from keras import backend as keras

        self.base = base
        self.model, self.preprocess = self.load_model()
        self.batch_size = batch_size
        self.data_format = keras.image_data_format()

    def load_model(self):
        if self.base == "resnet50":
            from keras.applications.resnet50 import (
                ResNet50,
                preprocess_input,
            )

            cnn_base_model = ResNet50(
                include_top=False, weights="imagenet", pooling="avg"
            )
        elif self.base == "vgg16":
            from keras.applications.vgg16 import VGG16, preprocess_input

            cnn_base_model = VGG16(include_top=False, weights="imagenet", pooling="avg")
        elif self.base == "inception_v3":
            from keras.applications.inception_v3 import (
                InceptionV3,
                preprocess_input,
            )

            cnn_base_model = InceptionV3(
                include_top=False, weights="imagenet", pooling="avg"
            )
        elif self.base == "xception":
            from keras.applications.xception import (
                Xception,
                preprocess_input,
            )

            cnn_base_model = Xception(
                include_top=False, weights="imagenet", pooling="avg"
            )
        else:
            raise ValueError(f"{self.base} is not a valid model")
        return cnn_base_model, preprocess_input

    def predict(self, x):
        from keras import backend as keras

        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = self.preprocess(x.astype(keras.floatx()))
        return self.model.predict(x, batch_size=self.batch_size, verbose=False)
