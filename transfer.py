from keras.models import Sequential
from keras.layers import Input,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
in_img = Input(shape=(3, 32, 32))

def gen_model():
    in_img = Input(shape=(3, 32, 32))
    x = Convolution2D(12, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(in_img)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    x = Convolution2D(3, 1, 1, border_mode='valid', name='conv2')(x)
    x = Activation('relu', name='relu_conv2')(x)
    x = GlobalAveragePooling2D()(x)
    o = Activation('softmax', name='loss')(x)
    model = Model(input=in_img, output=[o])
    return model

#parent model
model=gen_model()
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

#saving model weights
model.save('model_weights.h5')

#loading weights to second model
model2=gen_model()
model2.compile(loss="categorical_crossentropy", optimizer="adam")
model2.load_weights('model_weights.h5', by_name=True)

model2.layers.pop()
model2.layers.pop()
model2.summary()

#editing layers in the second model and saving as third model
x = MaxPooling2D()(model2.layers[-1].output)
o = Activation('sigmoid', name='loss')(x)
model3 = Model(input=in_img, output=[o])

model.summary()
model.layers.pop()
model.layers.pop()
model.summary()

x = BatchNormalization()(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size=8)(x)
y = Flatten()(x)
outputs = Dense(num_classes,
                activation='softmax',
                kernel_initializer='he_normal')(y)
x = MaxPooling2D()(model.layers[-1].output)
o = Activation('sigmoid', name='loss')(x)

model2 = Model(input=in_img, output=[o])
model2.summary()
