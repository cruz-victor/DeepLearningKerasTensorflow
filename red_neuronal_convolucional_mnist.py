#1. IMPORTACION Y DEFINICION DE LIBRERIAS
import tensorflow as tf
import numpy as np
from keras.datasets import mnist #Base de datos de imagenes de numeros escritos a manos.
from keras.models import Sequential, Model #Modelos
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input  #Capas. Conjunto de neuronas.
from keras import backend as K #Operaciones que requieran tensorflow. Librerias que se relacionan con tensorflow.
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

#2. DEFINICION DE PARAMETROS Y PREPROCESADOS DE LA BD
batch_size=100 #Numero de imagenes que va a ver el sistema por cada iteracion.#Por cada iteracion se va tomar 100 imagenes de numeros escritos.
num_classes=10 #Hay 10 numeros=clases. #Mnist tiene numeros escritos del 0-9.
epochs=10 #Numero de veces que va a recorrer la base de datos.

filas, columnas=28,28 #Tamano de las imagenes de 28x28 de la base de datos.#28 filas, 28 columnas, 1 canal.

(x_train, y_train), (x_test, y_test)=mnist.load_data() #Cargar la base de datos. (x_train,y_train) son datos de entrenamiento. (x_test,y_test) son datos de prueba

#Preprocesar los datos
x_train=x_train.reshape(x_train.shape[0], filas, columnas, 1)#reshape, permite convertir todas las imagenes de entrenamiento a un tamano de 28x28 pixeles y un canal(gris).
x_test=x_test.reshape(x_test.shape[0], filas, columnas, 1)#reshape, permite convertir todas las imagenes de prueba a un formato 28x28, gris. x_train.shape[0] (60000,28,28,1)

#Convertir la imagen a float32
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#Normalizar los datos (X)
x_train= x_train / 255 #Escalar las imagenes (de 8 bits (0-255) y de un solo canal) de entenamiento entre 0 y 1. #Evita saturaciones en las funciones de activacion.
x_test= x_test / 255

#Convertir a formato categorico (Y)
#Convertir los valores enteros de etiquetas (0...9) a valores de array
#Ej.:
    # 1->[0,1,0,0,0,0,0,0,0,0]
    # 9->[0,0,0,0,0,0,0,0,0,1]
y_train=to_categorical(y_train, num_classes) #No se puede introducir un numero (1,2,3,4,..) convertir a formato de etiquetas.
y_test=to_categorical(y_test, num_classes)


#3. CREACION DEL MODELO (RNA) Y COMPILACION DEL MISMO
print("###########3. CREACION DEL MODELO (RNC) Y COMPILACION DEL MISMO")

#Con Model, hay entradas y salidas de capas.
Entradas=Input(shape=(28,28,1)) #Capa de entrada #Tensor de entrada.
#---Etapa de extraccion de caracteristicas
x=Conv2D(64, kernel_size=(3,3),activation='relu')(Entradas)#'x'=salida de la Capa Conv. 'Entradas'=entrada de la Capa Conv.#64 filtros. #Ej. encontrar bordes
x=Conv2D(128, kernel_size=(3,3),activation='relu')(x)#'x'=salida de la Capa Conv. 'x'=entrada de la Capa Conv.#Con mas capas convoluciones, mayor numero de filtros. #Ej. encontrar esquinas, sombras
x=MaxPooling2D(pool_size=(2,2))(x)#Obtener las caracteristicas mas relevantes.
#---Etapa de clasificacion
x=Flatten()(x) #Convertir a una dimension
x=Dense(68,activation='relu')(x)#
x=Dropout(0.25)(x)
x=Dense(20,activation='relu')(x)
x=Dropout(0.25)(x)
x=Dense(num_classes, activation='softmax')(x)

modelo=Model(inputs=Entradas, outputs=x)
modelo.summary()#Resumen por capas.

#modelo.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.categorical_crossentropy,metrics=['categorical_accuracy'])
print('###########3.1 COMPILACION DEL MODELO')
modelo.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])#Compilar el modelo.
#optimizer='adam', algoritmo de optimizacion. con parametro por defecto.
#loss='categorical_crossentropy', clasificar multiclase. Se requiere clasificar numeros del 0-9
#metrics=['categorical_accuracy'], las funciones de perdida, normalmente no dan una idea de como entrena el modelo. Metric permite ver de forma intuitiva como esta entrenando el modelo.
    #categorical_accuracy, para multiclases.
    #acuracy, para clasificacion binaria.

#4. ENTRENAMIENTO Y EVALUACION
print('###########4. ENTRENAMIENTO Y EVALUACION')
modelo.fit(x=x_train, y=y_train,batch_size= batch_size,epochs=epochs, validation_data=(x_test, y_test), verbose=1)#verbose, muesra en consola los mensajes del procesamiento
puntuacion=modelo.evaluate(x_test, y_test, verbose=1)#Evalua el modelo a partir de los datos de prueba.
print('###########4.1 PUNTUACION')
print(puntuacion)
