import streamlit as st
from keras.applications import MobileNetV2
from keras.applications import InceptionResNetV2
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


@st.cache
def feature_extraction(cnn_name, image_size, img):
    print('saeid')

    if cnn_name == 'MobileNetV2':
        model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))

    elif cnn_name == 'ResNet50':
        model = ResNet50(include_top=False, weights='imagenet',
                         input_shape=(image_size[0], image_size[1], 3))
    elif cnn_name == 'InceptionResNetV2':
        model = InceptionResNetV2(include_top=False, weights='imagenet',
                                  input_shape=(image_size[0], image_size[1], 3))

    out_lay = GlobalAveragePooling2D()(model.output)

    # out_lay = Flatten()(model.output)

    model = Model(inputs=model.input, outputs=out_lay)

    return model.predict(img)
