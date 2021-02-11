#1. IMPORTACION Y DEFINICION DE LIBRERIAS
import tensorflow as tf
import numpy as np
from keras.datasets import mnist #Base de datos de imagenes de numeros escritos a manos.
from keras.models import Sequential #Modelos
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D #Capas
from keras import backend as k #Operaciones que requieran tensorflow
import keras
from keras.utils.np_utils import to_categorical


#2. DEFINICION DE PARAMETROS Y PREPROCESADOS DE LA BD
batch_size=100 #Numero de imagenes que va a ver el sistema por cada iteracion.
num_classes=10 #Hay 10 numeros=clases.
epochs=10 #Numero de veces que va a  recorrer la base de datos.

filas, columnas=28,28 #Tamano de las imagenes de 28x28.

(x_train, y_train), (x_test, y_test)=mnist.load_data() #Cargar la base de datos. (xt,yt) son datos de entrenamiento. (xtest,ytest) son datos de prueba

#Preprocesar los datos
x_train=x_train.reshape(x_train.shape[0], filas, columnas, 1)#reshape, permite asignar normas y formas. #Es mejor trabajar con escala de grises.
x_test=x_test.reshape(x_test.shape[0], filas, columnas, 1)#x_train.shape[0] (60000,28,28,1)

#Convertir la imagen a float32
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#Normalizar los datos (X)
x_train= x_train / 255
x_test= x_test / 255

#Convertir a formato categorico (Y)
#Convertir los valores enteros de etiquetas (0...9) a valores de array
#Ej.:
    # 1->[0,1,0,0,0,0,0,0,0,0]
    # 9->[0,0,0,0,0,0,0,0,0,1]
y_train=to_categorical(y_train, num_classes)
y_test=to_categorical(y_test, num_classes)


#3. CREACION DEL MODELO Y COMPILACION DEL MISMO
modelo=Sequential()
modelo.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))#Capa convolucional
modelo.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))#Capa convolucional con mayor profundidaa
modelo.add(MaxPooling2D(pool_size=(2,2)))
modelo.add(Flatten()) #Reduce las dimensiones a una sola dimension
modelo.add(Dense(68))
modelo.add(Dropout(0.25))
modelo.add(Dense(20))
modelo.add(Dropout(0.25))
modelo.add(Dense(num_classes,activation='softmax'))

#modelo.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.categorical_crossentropy,metrics=['categorical_accuracy'])
modelo.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

#4. ENTRENAMIENTO Y EVALUACION
modelo.fit(x_train, y_train, batch_size, epochs, validation_data=(x_test, y_test), verbose=1)#verbose, muesra en consola los mensajes del procesamiento
puntuacion=modelo.evaluate(x_test, y_test, verbose=1)
print(puntuacion)
