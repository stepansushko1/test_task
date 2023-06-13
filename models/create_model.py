import tensorflow as tf

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import UpSampling2D,Conv2D,BatchNormalization,MaxPool2D,ReLU,AveragePooling2D
from tensorflow.keras.layers import concatenate

def add_upsample(model, previous_layer, encoder_layer):

    prev_activations = previous_layer.output
    try:
        encoder_activations = encoder_layer.output
    except Exception:
        encoder_activations = encoder_layer

    num_filters = round(prev_activations.shape[-1] / 2)
    up = UpSampling2D()(prev_activations)  # double height and width

    conv_1 = Conv2D(num_filters, 3, padding='same', activation='relu')(concatenate([up, encoder_activations]) )
    conv_2 = Conv2D(num_filters, 3, padding='same', activation='relu')(conv_1)
    return Model(inputs=model.input, outputs=conv_2)


def bottle_neck(prev_max_pool, n_filters):

    conv = Conv2D(n_filters, 3, padding='same')(prev_max_pool)
    normalised = BatchNormalization()(conv)
    activ = ReLU()(normalised)
    conv2 = Conv2D(n_filters, 3, padding='same')(activ)
    normalised2 = BatchNormalization()(conv2)
    activ2 = ReLU()(normalised2)

    return activ2

def U_NET(input_shape: tuple[int, ...], weights=None) -> Model:
    """Creates model with UNet architecture"""
    inp = Input(input_shape, dtype=tf.float32)
    x = BatchNormalization()(inp)

    matchings_skip = []  # we'll store activations here

    for n_filters in (32, 64, 128, 256):
        conv = Conv2D(n_filters, 3, padding='same')(x)
        normalised = BatchNormalization()(conv)
        activ = ReLU()(normalised)
        conv2 = Conv2D(n_filters, 3, padding='same')(activ)
        normalised2 = BatchNormalization()(conv2)
        activ2 = ReLU()(normalised2)
        matchings_skip.append(activ2)
        x = MaxPool2D()(activ2)


    model = Model(inputs=inp, outputs=bottle_neck(x, 512))


    for layer in reversed(matchings_skip):
        model = add_upsample(model, model.layers[-1], layer)

    final_conv = Conv2D(1, 1, activation='sigmoid')(model.layers[-1].output)
    model = Model(inputs=model.input, outputs=final_conv)

    if weights:
        model.load_weights(weights)
    return model


def U_NET_fullresolution(trained_unet, trained=False):
    """
    Full resolution unet. Takes trained UNET as input
    """
    model = Sequential()
    model.add(AveragePooling2D((3, 3), input_shape=(768, 768, 3)))
    if trained == False:
        model.add(trained_unet)
    if trained == True:
        model.add(U_NET((256, 256, 3)))
    model.add(UpSampling2D((3, 3)))
    return model