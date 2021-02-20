import sys
import json
import codecs
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Input,Bidirectional,GlobalMaxPool1D,Flatten,BatchNormalization,LeakyReLU
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence,text
from keras import Model
from keras.models import load_model
from keras.datasets import imdb #Dataset para el analisis de sentimiento
import matplotlib.pylab as plt

#Variables de entrada
batch_size = 32
epochs = 30
tamano_embedding = 128 #Es capa especial para el procesado de texto, asigna a las palabras un vectorizacion unica.
maximas_caracteristicas=20000 #Maximo de palabras caracteristicas que va a tener el dataset.
maxima_longitud=80

#Cargado de bases  de datos
#xentrenamiento no son palabras, son numeros. Imdb ya esta convertido a numeros.
#En sklearn se convierto las letras a numeros.
(xentrenamiento,yentrenamiento),(xtest,ytest)=imdb.load_data(num_words=maximas_caracteristicas)
xentrenamiento=sequence.pad_sequences(xentrenamiento,maxlen=maxima_longitud)
xtest=sequence.pad_sequences(xtest,maxlen=maxima_longitud)

#Creacion de modelo
entrada = Input(shape=(maxima_longitud, ))
x = Embedding(maximas_caracteristicas, tamano_embedding)(entrada) #Capa especial para texto.
x = LSTM(tamano_embedding, return_sequences=True,activation='relu')(x) #returns_sequences devuelde los estados obtenidos por embedding.
x = Flatten()(x) #Llevar a una dimension
x = Dense(1, activation="sigmoid",kernel_initializer='zeros',bias_initializer='zeros')(x)
modelo = Model(inputs=entrada, outputs=x)
modelo.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
modelo.summary()

#Clasificacion binaria, positivo o negativo

#entrenamiento
#Callback para guardar el mejor modelo de las mejores epocas.
checkpoint = ModelCheckpoint('deteccion_texto.h5', monitor='val_binary_accuracy', verbose=1, save_best_only=True,save_weights_only=False, mode='auto')
history=modelo.fit(xentrenamiento, yentrenamiento, batch_size=batch_size, epochs=5, callbacks=[checkpoint],validation_data=(xtest,ytest), shuffle=True, verbose=1)

#visualizacion de resultaoos
plt.figure(1)
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Perdidas del Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Test','Entrenamiento'], loc='upper left')

plt.figure(2)
plt.plot(history.history['val_binary_accuracy'])
plt.plot(history.history['binary_accuracy'])
plt.title('Precision del Modelo')
plt.ylabel('Precision')
plt.xlabel('Epocas')
plt.legend(['Test','Entrenamiento'], loc='upper left')

#Scores
#Salida de la red en validacion
salida=modelo.predict(xentrenamiento[round(len(xentrenamiento[:,0])*0.9):round(len(xentrenamiento[:,0])),:])

#Binarizacion.
salida[salida<0.5]=0
salida[salida>=0.5]=1

#Obtener la DIFERENCIAS respecto a los datos reales.
diferencia=abs(yentrenamiento[round(len(xentrenamiento[:,0])*0.9):round(len(xentrenamiento[:,0]))]-np.uint8(salida[:,0]))
#Las difernecias > 0 implican que no se ha acertado en ese punto.
puntos=diferencia[diferencia>0]
#Punto acertado y no acertados
score=(1-(len(puntos)/len(diferencia)))

print('La mejor puntuacion es %f'% (score))
plt.show()
