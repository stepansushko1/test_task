import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy


def dice_coef(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    beta: float = 0.5,
    smooth: float = 1.0,
):
    """
    dice coeficient metric. Should increase
    """
    intersection = K.sum(y_true * y_pred, axis=range(1, K.ndim(y_pred)))

    union = K.sum((
        y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    ), axis=range(1, K.ndim(y_pred)))
    return (intersection + smooth) / (union + smooth)


def cross_entropy_dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    beta: float = 0.5,
    smooth: float = 1.0,
):
    """
    Combination of binary cross entropy and dice losses. Should decrease
    """
    bce = binary_crossentropy(y_true, y_pred)

    bce = K.mean(bce, axis=range(1, K.ndim(bce)))

    dice_coefficient = dice_coef(y_true, y_pred, 0.5, smooth)


    return beta * (1.0 - dice_coefficient) + (1.0 - beta) * bce
