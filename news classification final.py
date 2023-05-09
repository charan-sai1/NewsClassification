#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras      #Importing the libraries.
from tensorflow import keras as tk
from keras.utils import np_utils as nu
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.ops.variables import trainable_variables as trv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten


# In[ ]:


(X_train, Y_train), (X_test, Y_test) = tk.datasets.cifar10.load_data() #Spliting the dataset.


# In[ ]:


print('Number of Training Images are : {}'.format(X_train.shape))  #Printing Image Information.
print('Number of Testing Images are : {}'.format(X_test.shape))


# In[ ]:


for i in range(234,238):   #Viewing the Images.
  plt.subplot(120+ 1 + i)
  img=X_train[i]
  plt.imshow(img)
  plt.show()


# In[ ]:


X_train=X_train.reshape(X_train.shape[0],32,32,3)   #Reshaping the Images.
X_test=X_test.reshape(X_test.shape[0],32,32,3)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train /=255
X_test /=255
n_classes =10
print("shape before one-hot encoding:",Y_train.shape)
Y_train=nu.to_categorical(Y_train, n_classes)
Y_test=nu.to_categorical(Y_test, n_classes)
print("shape after one-hot encoding:",Y_train.shape)


# In[ ]:


model = Sequential()   #Building the model

#Adding convolutional layers
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(75, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')) 
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout (0.25))
model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) 
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

#Adding hidden Layer
model.add (Dense(500, activation='relu')) 
model.add (Dropout (0.4))
model.add (Dense(250, activation='relu'))
model.add(Dropout(0.3))

#Adding output Layer
model.add(Dense(10, activation='softmax'))

#Compiling
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#Training the model 
model.fit(X_train, Y_train, batch_size=128, epochs=50, validation_data=(X_test,Y_test))


# In[ ]:


classes=range(0,10)     #Naming the classes.
names= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
clabels= dict(zip(classes,names))
batch=X_test[100:109]
labels=np.argmax(Y_test[100:109],axis=-1)
predictions=model.predict(batch, verbose=1)


# In[ ]:


for image in predictions:  #Sum of accuracy.
  print(np.sum(image))


# In[ ]:


cresult=np.argmax(predictions,axis=-1)   #Setting the table
print(cresult)


# In[ ]:


fig, axs=plt.subplots(3, 3, figsize=(20,10))   #Testing the model generated.
fig.subplots_adjust(hspace=1)
axs=axs.flatten()


for i, img in enumerate(batch):
  for key, value in clabels.items():
    if cresult[i] == key: 
      title='Prediction: {}\nActual: {}'.format(clabels[key], clabels[labels [i]]) 
      axs[i].set_title(title)
  #Plottiing the image 
  axs[i].imshow(img)

#Show the plot
plt.show()

