import tensorflow as tf
import numpy as np
import keras
import math
from pandas import read_csv
from keras.datasets import cifar10
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Input,Conv3D,BatchNormalization,ConvLSTM2D,LSTM,GRU
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD,Adam
from keras.preprocessing import image
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler #sklearn, libreria de machine learning. MinMaxScaler es una libreria para el escalado.
from sklearn.metrics import mean_squared_error


# Conversion Array-Matriz
#Teniendo el cuenta la variable mirar_atras, obtener los valores anteriores que tiene que tener en cuenta la red.
def create_base_datos(datos, mirar_atras=1):
	datosX, datosY = [], []
	for i in range(len(datos)-mirar_atras-1):
		a = datos[i:(i+mirar_atras), 0]
		datosX.append(a)
		datosY.append(datos[i + mirar_atras, 0])
	return np.array(datosX), np.array(datosY)


#Cargar Datos
dataframe = read_csv('C:\\Users\\vic\\Documents\\Victor Cruz Gomez Windows 10\\CursoDeepLearnigKerasTensorflow\\base_datos_pasajeros\\international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
datos = dataframe.values
datos = datos.astype('float32')

#Normalizacion de Datos
escalado = MinMaxScaler(feature_range=(0, 1)) #Normalizando con libreria de sklearn Que la escala este entre 0 y 1
datos = escalado.fit_transform(datos) #Transformar los datos a la escala definida.

#Division en Train/Test
tsize = int(len(datos) * 0.67) #67% para el entrenamienot
testsize = len(datos) - tsize #23% de los datos para el test
entrenamiento, test = datos[0:tsize,:], datos[tsize:len(datos),:]#Utilizando los tamanos obtenemos los datos de entrenamiento y test

# Cambio de tamaños de train y test
mirar_atras= 1 #Cantidad de valores previos a tener en cuenta. Ej.: Quiero saber lo que ha ocurrido desde hace 5 momentos?
tX, tY = create_base_datos(entrenamiento, mirar_atras) #Obtiene los x valores anteriores del entrenamiento
testX, testY = create_base_datos(test, mirar_atras) #Obtiene los x valores de anteriores del test
tX = np.reshape(tX, (tX.shape[0], 1, tX.shape[1])) #Por la forma que tienen de obtener los datos las LSTM. Pasos temporales al final de todos los valores.
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Creacion de Red con LSTM
input=Input(shape=(1,mirar_atras))
#x=LSTM(10)(input) #Unidad LSTM
x=GRU(10)(input) #Unidad GRU. Lleva una celda que tiene 10 unidades de GRU con las que va hacer la prediccion y encadenarlas.
x=Dense(1)(x)
model=Model(inputs=input,outputs=x)
model.compile(loss='mean_squared_error', optimizer='adam') #Es un problema de regresion=mena_squared_error
history=model.fit(tX, tY, epochs=100, batch_size=1, verbose=2)#Entrenamiento.
#epochs=100, por que el dataset es muy pequeno.por que el dataset es muy pequeno.
#batch_size=1 por que el dataset es muy pequeno.

#Grafico para observar la prediccion en el set de test
plt.figure(1)
plt.plot(history.history['loss'])
plt.title('Perdidas del Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento'], loc='upper left')


# Realizacion de predicciones
prediciones_entrenamiento = model.predict(tX)
prediciones_test = model.predict(testX)

# Inversion de las predicciones para calcular su error
#Desacer la normalizacion para observar los valores antes de la normalizacion.
prediciones_entrenamiento = escalado.inverse_transform(prediciones_entrenamiento)
tY = escalado.inverse_transform([tY])
prediciones_test = escalado.inverse_transform(prediciones_test)
testY = escalado.inverse_transform([testY])

# Calculo de la raiz del error cuadratico medio o RMSE
Puntuacion_Train = math.sqrt(mean_squared_error(tY[0,:], prediciones_entrenamiento[:,0]))
Puntuacion_Test = math.sqrt(mean_squared_error(testY[0], prediciones_test[:,0]))
print('Puntuacion Train: %.2f RMSE  y Puntuacion Test: %.2f RMSE' % (Puntuacion_Train,Puntuacion_Test))

#Ver como se ajustaron los datos en el entrenamiento y test
# Desplazamiento de predicciones de entrenamiento
plot_prediccion_entrenamiento = np.empty_like(datos)
plot_prediccion_entrenamiento[:, :] = np.nan
plot_prediccion_entrenamiento[mirar_atras:len(prediciones_entrenamiento)+mirar_atras, :] = prediciones_entrenamiento

# Desplazamiento de predicciones de test
plot_prediccion_test = np.empty_like(datos)
plot_prediccion_test[:, :] = np.nan
plot_prediccion_test[len(prediciones_entrenamiento)+(mirar_atras*2)+1:len(datos)-1, :] = prediciones_test

# Mostrar predicciones y datos
plt.figure(2)
plt.plot(escalado.inverse_transform(datos))
plt.plot(plot_prediccion_entrenamiento)
plt.plot(plot_prediccion_test)
plt.show()



