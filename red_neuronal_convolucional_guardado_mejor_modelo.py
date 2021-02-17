import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Input
from keras import backend as K
from keras.callbacks import TensorBoard

batch_size=100
num_classes=10
epochs=3

filas,columnas= 28,28

(x_train,y_train),(x_test,y_test)= mnist.load_data()

x_train=x_train.reshape(x_train.shape[0],filas,columnas,1)
x_test=x_test.reshape(x_test.shape[0],filas,columnas,1)

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

x_train=x_train/255
x_test=x_test/255

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)




Entradas=Input(shape=(28,28,1))
x=Conv2D(64,kernel_size=(3,3),activation='relu')(Entradas)
x=Conv2D(128,kernel_size=(3,3),activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Flatten()(x)
x=Dense(68,activation='relu')(x)
#x=Dropout(0.25)(x)
x=Dense(20,activation='relu')(x)
#x=Dropout(0.25)(x)
x=Dense(num_classes,activation='softmax')(x)

modelo = Model(inputs=Entradas, outputs=x)
modelo.summary()

#---GUARDADO DE LOS PESOS JUNTO A SU ARQUITECTURA---
#monitor='val_loss', fijarse en una variables.
    #Normlamente fijarse en el 'loss de validacion' o una metrica de validacion que exprese lo bueno del modelo
#verbose=1, ver todos los pasos en en consola.
#save_best_only=False, guardar todos los mejores modelos que vaya dentectando a lo largo del entrenamiento (False).
#save_weights_only=False, guardado de todos los mejores pesos que vaya detectando
#mode='auto', detecta si los valores de 'val_loss' sube o baja.
    #Monitorizar pedidas, se requiera que en la epoca actual tengamos menos perdidas que el anterior. Mejorando las perdidas
    #Accuray, se quire que la precision suba en la epoca actual
    #Modo auto, detecta si tiene que hacerlo por 'val_loss' o 'accuracy'
#period=1
autoguardado=keras.callbacks.ModelCheckpoint('automodelo.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#---GUARDADO SOLO DE LOS PESOS---
autoguardado2=keras.callbacks.ModelCheckpoint('autopesos.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)

#En keras, es mejor guardar los pesos. Los pesos simplemente son numeros.
#Guardar el modelo con su arquitectura puede producir problemas por la version de keras y tensorflow.
#El guardado del modelo y pesos se realiza al final de cada epoca.
#Cuando empiece a sobreentrenar dejara de guargar el modelo y pesos.
#En los callback se tienen guardado el ultimo modelo eficiente.

modelo.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['categorical_accuracy'])

#---CALLBACKS---
#callbacks=[autoguardado,autoguardado2], permite guardar el modelo y los pesos a lo largo del entrenamiento.
modelo.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,callbacks=[autoguardado,autoguardado2],verbose=1,validation_data=(x_test,y_test))

puntuacion=modelo.evaluate(x_test,y_test,verbose=1)

print(puntuacion)

#---GUARDADO AL FINAL DEL ENTRENAMIENTO---
#Otra forma de guardar el modelo y su arquitectura.
#Solo se guarda el ultimo modelo entrenado en el fit.
modelo.save('modelo3.h5')
#---CARGADO DEL MODELO Y ARQUITECTURA
from keras.models import load_model
modelo2=load_model('modelo3.h5')

#---CARGADO DEL LOS PESOS
#Al cargado de los pesos, el modelo tiene que estar preestablecido.
modelo.load_weights('autopesos.h5')
#---CARGADO DEL MODELO Y PESOS MAS OPTIMO---
modelo2=load_model('automodelo.h5')