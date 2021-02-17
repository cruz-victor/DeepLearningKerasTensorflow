import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar100
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Input
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD,Adam
from keras.applications import VGG16
import matplotlib.pylab as plt

batch_size=100
num_classes=100
epochs=5

(x_train,y_train),(x_test,y_test)= cifar100.load_data()

_,filas, columnas, canales = x_train.shape #32x32x3

#Convertir el formato a float32
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#Normalizacion
x_train=x_train/255
x_test=x_test/255

#Categorizar (Codificacion OneHot)
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)


Entradas=Input(shape=(filas,columnas,canales))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(Entradas)
#x=Dropout(0.25)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
#x=Dropout(0.25)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), name='block2_pool')(x)

#Fully Conected/RNA
x=Flatten()(x)
x=Dense(512,activation='relu')(x)
x=Dropout(0.5)(x) #Regularizar para evitar el overffiting. Hay 512 neuronas. Dropoout evitara el overfitting
x=Dense(num_classes,activation='softmax')(x) #Softmax obtiene una probabilidad en 0% y 100%

modelo = Model(inputs=Entradas, outputs=x)
#modelo.summary()

Adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.9)#SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

modelo.compile(loss=keras.losses.categorical_crossentropy,optimizer=Adam,metrics=['categorical_accuracy'])

history=modelo.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))#history, saca el historial del entrenamiento, como ha ido variando por epocas.

puntuacion=modelo.evaluate(x_test,y_test,verbose=1)

print(puntuacion)

#Graficando el historial del entrenamiento
plt.figure(1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Precision de Modelo')
plt.ylabel('Precision')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')


plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perdidas del Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()
