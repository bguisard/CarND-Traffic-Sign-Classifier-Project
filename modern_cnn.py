"""
    This is a more modern approach to convnets, featuring many of the state-of-the-art concepts
    introduced over the past few years.

    1 - Xavier weight initialization
    http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    2 - Batch Normalization
    https://arxiv.org/pdf/1502.03167.pdf

    3 - Dropout
    http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf

    4 - No Max Pooling - higher stride conv is used to avoid information loss
    https://arxiv.org/pdf/1412.6806.pdf

"""

from keras.layers import Convolution2D, Dropout, BatchNormalization, Flatten, Dense, Input, Activation
from keras.models import Model


def modern_cnn(n_classes=10, img_width=32, img_height=32, color_channels=3):

    img_input = Input(shape=(img_width, img_height, color_channels), name='input_img')

    # Block 1:
    x = Convolution2D(32, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_1a')(img_input)
    x = BatchNormalization(name='bn_1a')(x)
    x = Activation('relu', name='relu_1a')(x)
    x = Convolution2D(32, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_1b')(x)
    x = BatchNormalization(name='bn_1b')(x)
    x = Activation('relu', name='relu_1b')(x)
    x = Convolution2D(32, 2, 2,
                      border_mode='valid',
                      activation='relu',
                      init='glorot_normal',
                      subsample=(2, 2),
                      name='conv_pool_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)

    # Block 2:
    x = Convolution2D(64, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_2a')(x)
    x = BatchNormalization(name='bn_2a')(x)
    x = Activation('relu', name='relu_2a')(x)
    x = Convolution2D(64, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_2b')(x)
    x = BatchNormalization(name='bn_2b')(x)
    x = Activation('relu', name='relu_2b')(x)
    x = Convolution2D(64, 2, 2,
                      border_mode='valid',
                      activation='relu',
                      init='glorot_normal',
                      subsample=(2, 2),
                      name='conv_pool_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)

    # Block 3:
    x = Convolution2D(128, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_3a')(x)
    x = BatchNormalization(name='bn_3a')(x)
    x = Activation('relu', name='relu_3a')(x)
    x = Convolution2D(128, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_3b')(x)
    x = BatchNormalization(name='bn_3b')(x)
    x = Activation('relu', name='relu_3b')(x)
    x = Convolution2D(128, 2, 2,
                      border_mode='valid',
                      activation='relu',
                      init='glorot_normal',
                      subsample=(2, 2),
                      name='conv_pool_3')(x)
    x = Dropout(0.2, name='dropout_3')(x)

    # Fully Connected:
    x = Flatten(name='flatten_4')(x)
    x = Dense(512, activation='relu', name='fc_6')(x)
    x = Dropout(0.5, name='dropout_7')(x)
    x = Dense(n_classes, activation='softmax', name='logits')(x)

    return Model(img_input, x, name='modern')
