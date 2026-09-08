"""Core U-Net used by the DL-BCN experiments.

The model accepts a square multichannel predictor field and returns one
continuous refined concentration field. During training, ``y_true`` stores the
target in channel 0, the real-station training mask in channel 1, and the
pseudo-station training weight in channel 2.
"""

import tensorflow as tf
from keras import Model, layers


def build_unet(input_size=128, input_channels=8):
    """Build the shallow two-level U-Net used in the reported experiments."""

    def double_conv(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return layers.Conv2D(filters, 3, padding="same", activation="relu")(x)

    inputs = layers.Input((input_size, input_size, input_channels))

    encoder_1 = double_conv(inputs, 32)
    pooled_1 = layers.MaxPool2D()(encoder_1)
    encoder_2 = double_conv(pooled_1, 64)
    pooled_2 = layers.MaxPool2D()(encoder_2)

    bottleneck = layers.Conv2D(128, 3, padding="same", activation="relu")(pooled_2)
    bottleneck = layers.Dropout(0.2)(bottleneck)
    bottleneck = layers.Conv2D(128, 3, padding="same", activation="relu")(bottleneck)

    decoder_2 = layers.UpSampling2D(2)(bottleneck)
    decoder_2 = layers.concatenate([decoder_2, encoder_2])
    decoder_2 = double_conv(decoder_2, 64)
    decoder_1 = layers.UpSampling2D(2)(decoder_2)
    decoder_1 = layers.concatenate([decoder_1, encoder_1])
    decoder_1 = double_conv(decoder_1, 32)

    outputs = layers.Conv2D(1, 1, activation="linear")(decoder_1)
    return Model(inputs, outputs, name="DL_BCN_UNet")


def _masked_mean(values, weights):
    weights = tf.cast(weights, tf.float32)
    return tf.reduce_sum(values * weights) / (tf.reduce_sum(weights) + 1e-6)


def group_mean_mse(pseudo_relative_weight=0.25):
    """Return the real/pseudo group-mean MSE used for model optimization.

    A value of 0.25 gives normalized group weights of 0.8 for real targets and
    0.2 for pseudo targets. If no pseudo target is present, the loss reduces to
    the real-station mean MSE.
    """
    real_weight = 1.0 / (1.0 + pseudo_relative_weight)
    pseudo_weight = pseudo_relative_weight / (1.0 + pseudo_relative_weight)

    def loss(y_true, y_pred):
        squared_error = tf.square(y_true[..., 0] - y_pred[..., 0])
        real_mask = y_true[..., 1]
        pseudo_mask = y_true[..., 2]
        real_mean = _masked_mean(squared_error, real_mask)
        pseudo_mean = _masked_mean(squared_error, pseudo_mask)
        has_pseudo = tf.cast(tf.reduce_sum(pseudo_mask) > 0, tf.float32)
        grouped = real_weight * real_mean + pseudo_weight * pseudo_mean
        return has_pseudo * grouped + (1.0 - has_pseudo) * real_mean

    return loss


if __name__ == "__main__":
    model = build_unet()
    model.summary()
