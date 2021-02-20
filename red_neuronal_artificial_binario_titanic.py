from keras.layers import Input, Flatten, Dense, Dropout, Reshape, PReLU,LeakyReLU,BatchNormalization,Activation
from keras.models import Model
import numpy as np
import keras
import keras.layers as layers
import keras.backend as K
import math
from keras.models import load_model,Model
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


#---BASE DE DATOS EXTERNA
traindata = pd.read_csv('/home/dhome/CURSO_KERAS_DAVID/Sesion_6/titanic_database/train.csv')

testdata = pd.read_csv('/home/dhome/CURSO_KERAS_DAVID/Sesion_6/titanic_database/test.csv')

#---PROCESADO DE DATOS
def process_data(datos):
    xt = datos[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']] #Obtener los 6 datos.
    xt = xt.fillna(xt.mean())#Eliminar los valores NA, valores sin sentido.

    xt.loc[:,'Sex'] = xt['Sex'].replace(['female','male'],[0,1]).values#Cambiar las etiquetas de sexo por 0 y 1.
    xt.loc[:,'Pclass'] = xt.loc[:,'Pclass'] - 1 #Restar a un para cuadrarlo respecto a los valores normalizados introduccidos a la RN

    return xt

xt = process_data(traindata)

xtest = process_data(testdata)

#---CATEGORIZACION - ONEHOT
yt = traindata["Survived"] #Convertir el valor si sobrevivio o no a formato OneHot. Valore categorico
ytonehot = tf.keras.utils.to_categorical(yt, num_classes=2)
print(ytonehot)

#Convertir todos los datos de lista a array
arraytrainx=np.array(xt)
arraytrainy=np.array(yt)

#---MODELO DE RNA
Entradas = Input(shape=(6,))#6 parametros de entrada 'X'.
x=Dense(200)(Entradas)
x=BatchNormalization()(x)#Dejar los datos en media=0 y desviacion tipico en 1. Para que la escala no afecte al modelol
x=LeakyReLU(alpha=0.05)(x) #Activacion LeakyRelu, similar a la relu. Anade una pequena pendiente a la zona negativa de activaciones. Con alpha=0.0 se convertiria en relu. #La pendiente alfa permite mejorar la velocidad del entrenamiento y la convergencia del mismo.
#x=Dropout(0.15)(x)
x=Dense(50)(x) #Reducir el numero de neuronas, normalmente se reduce hasta llevar al valor de la salida.
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.05)(x)
#x=Dropout(0.1)(x)
x=Dense(1)(x)
x = Activation('sigmoid')(x) #Activacion sigmoidal por ser una clasificacion bianaria (sobrevio o no sobrevivio).

modelo = Model(inputs=Entradas, outputs=x)

#Callback, guarda el modelo y pesos durante el entrenamiento.
Guardado = keras.callbacks.ModelCheckpoint('titanic.h5', monitor='val_acc', verbose=0, save_best_only=True,save_weights_only=False, mode='auto', period=1)

Adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.9)#Optimizdor

modelo.compile(optimizer=Adam,loss='binary_crossentropy',metrics=['accuracy'])
#binary_crossentropy, es un problema binario.
#accuracy, es la precision binaria. La 'categorical_accuracy' es la precion categorica.
#accuracy, marca si el valor que queremos esta por arriba o debajo de 0.5.

history=modelo.fit(arraytrainx,arraytrainy ,epochs=400, batch_size=200, validation_split=0.2, callbacks=[Guardado], verbose=0) #No le damos un set de test. Que asigne un 20% de los datos a test.

#Visualizar la precion y perdida del modelo.
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Precision de Modelo')
plt.ylabel('Precision')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perdidas de Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()



