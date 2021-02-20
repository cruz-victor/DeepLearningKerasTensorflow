
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Input
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD,Adam
from keras.applications import VGG16
import matplotlib.pylab as plt
from keras.datasets import boston_housing

batch_size=10 #hay menos muestras.
epochs=50 #Porque hay menos muestras las epocas seran mas rapidas.

#---CARGADO DEL DATASET
#Predecir el precio de zonas en boston.
#Variable independiente: alumbrando, criminalidad, impuestos, etc.
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#---ESTANDARIZACION (no es normalizacion)
#Restar la media y dividir entre la desviacion estandar.
#El rango de datos sera:-1 a 1
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
print(x_train.shape[1])
x_test -= mean
x_test /= std

#Maximo de la salida regresada 'Y'.
#El escalo en Y se hace de otra manera, no hay problema.
maximo=np.max(y_train)
y_train=y_train/(maximo)
y_test=y_test/maximo

#Modelo clasico de RNA
Entradas=Input(shape=(13,)) #13 caracteristicas de 'X'
x=Dense(64,activation='relu')(Entradas)
x=Dense(64,activation='relu')(x)
x=Dense(1,activation='linear')(x) #activacion por ser un modelo de regresion.

modelo = Model(inputs=Entradas, outputs=x)
#modelo.summary()
Adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.9)#SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

modelo.compile(loss=keras.losses.mse,optimizer=Adam,metrics=['mse'])#Funcion de perdida='error minimo cuadratico'
#mse, compara si la salida se acerca al resultado.
#mae, error absoluto medio, otra funcion de perdida.
#No existe una metrica de la precision 'accuracy'.

history=modelo.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

puntuacion=modelo.evaluate(x_test,y_test,verbose=1)

print(puntuacion)


plt.figure(1)
plt.plot(np.sqrt(history.history['loss'])*maximo)
plt.plot(np.sqrt(history.history['val_loss'])*maximo)
plt.title('Perdidas de Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()

