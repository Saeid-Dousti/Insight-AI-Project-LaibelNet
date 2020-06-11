from keras.applications import MobileNetV2
from keras.applications import InceptionResNetV2
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


def cnn_model(ftr_ext, image_size):
    if ftr_ext == 0:
        model = MobileNetV2(include_top=False, weights='imagenet',
                            input_shape=(image_size[0], image_size[1], 3))
    elif ftr_ext == 1:
        model = ResNet50(include_top=False, weights='imagenet',
                         input_shape=(image_size[0], image_size[1], 3))
    elif ftr_ext == 2:
        model = InceptionResNetV2(include_top=False, weights='imagenet',
                         input_shape=(image_size[0], image_size[1], 3))

    out_lay = GlobalAveragePooling2D()(model.output)

    # out_lay = Flatten()(model.output)

    model = Model(inputs=model.input, outputs=out_lay)

    return model
