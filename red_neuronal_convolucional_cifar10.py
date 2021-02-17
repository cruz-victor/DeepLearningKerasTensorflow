import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import VGG16

batch_size = 100
num_classes = 10
epochs = 30

(x_train, y_train), (x_trainest, y_trainest) = cifar10.load_data()

# La imagen a colores tiene 3 canales
_, filas, columnas, canales = x_train.shape

x_train = x_train.astype('float32')
x_trainest = x_trainest.astype('float32')

x_train = x_train / 255
x_trainest = x_trainest / 255

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_trainest = tensorflow.keras.utils.to_categorical(y_trainest, num_classes)

Basica = 1
if (Basica == 1):
    #Modelo RNC basico.
    Entradas = Input(shape=(filas, columnas, canales))
    # Fase de extraccion de caracteristicas
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(Entradas)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Fase de clasificacion
    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

if (Basica == 0):
    #Modelo RNC UVGG16
    Entradas = Input(shape=(filas, columnas, canales))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(Entradas)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

modelo = Model(inputs=Entradas, outputs=x)
# Parametros, para problemas complejos
# Optimizador SGD
# lr, learning rate, valor bajo
# decay, en cada epoca baja el learning rate un poco
# momentum, valor por defecto.
# nesterov, que el algoritmo sea SGD nesterov, mejor que el SGD clasico
# Optimizador Adam=SGD+Momentum
# lr=0.001, learning rate
# Las betas deciden como adam va a tener los momentums
# beta_1=0.9,
# beta_2=0.9,
descenso_gradiente_estocastico = Adam(lr=0.001, beta_1=0.9,
                                      beta_2=0.9)  # SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
modelo.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=descenso_gradiente_estocastico,
               metrics=['categorical_accuracy'])

modelo.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_trainest, y_trainest))

puntuacion = modelo.evaluate(x_trainest, y_trainest, verbose=1)

print(puntuacion)