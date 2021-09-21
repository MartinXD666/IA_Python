# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 22:29:10 2021

@author: eltir
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importando Datos
sales_df = pd.read_csv("datos_de_ventas.csv")

#Visualizacion
sns.scatterplot(sales_df['Temperature'], sales_df['Revenue'])

#Creando set de entrenamiento
x_train = sales_df['Temperature']
y_train = sales_df['Revenue']

#Creando Modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.1) , loss='mean_squared_error')

#Entrenamiento
epochs_hist = model.fit(x_train, y_train, epochs = 1000)

keys = epochs_hist.history.keys()

#Grafico de Entrenamiento de Modelo
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de Perdidad durante Entrenamiento del Modelo')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend('Training Loss')

weights = model.get_weights()

#Prediccion
temp = 50
revenue = model.predict([temp])
print('La ganancia segun la Red Neuronal sera de: ', revenue)

#Grafico de Prediccion
plt.scatter(x_train, y_train, color='gray')
plt.plot(x_train, model.predict(x_train), color='red')
plt.ylabel('Ganancia [Dolares]')
plt.xlabel('Temperatura [gCelsius]')
plt.title('Ganancia Generada vs. Temperatura @Empresa de Helados')