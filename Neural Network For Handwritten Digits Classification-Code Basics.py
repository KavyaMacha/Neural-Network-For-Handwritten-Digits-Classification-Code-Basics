#!/usr/bin/env python
# coding: utf-8

# In[39]:


#importing libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd


# In[40]:


(x_train,y_train) ,(x_test,y_test)=keras.datasets.mnist.load_data()


# In[41]:


#train length
len(x_train)


# In[42]:


#test sample
len(x_test)


# In[43]:


#shape of the individual train sample
x_train[0].shape


# In[44]:


x_train[0]


# In[45]:


#plotting the first training image
plt.matshow(x_train[0])


# In[46]:


plt.matshow(x_train[1])


# In[47]:


plt.matshow(x_train[2])


# In[48]:


y_train[2]


# In[49]:


#y_train 
y_train[:5]


# In[50]:


#we need to flatten the data
#we use pandas to do that
x_train.shape


# In[51]:


x_train=x_train/255
x_test=x_test/255


# In[52]:


x_train_flattened=x_train.reshape(len(x_train),28*28)
x_test_flattened=x_test.reshape(len(x_test),28*28)


# In[53]:


print(x_train_flattened)
print(x_test_flattened)


# In[54]:


#now lets see the shape of x_test_flattend
x_test_flattened.shape


# In[55]:


x_train_flattened[0]


# In[56]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)


# In[58]:


#evalute the test dataset
model.evaluate(x_test_flattened,y_test)


# In[61]:


plt.matshow(x_test[0])


# In[62]:


y_pred=model.predict(x_test_flattened)


# In[ ]:





# In[63]:


y_pred[0]


# In[64]:


np.argmax(y_pred[0])


# In[65]:


y_pred_labels=[np.argmax(i) for i in y_pred]
y_pred_labels[:5]


# In[66]:


#confusion matrix
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)
cm


# In[68]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Truth")


# In[69]:


#after adding a hiddn layer
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)


# In[70]:


#evalute the test dataset
model.evaluate(x_test_flattened,y_test)


# In[71]:


y_pred=model.predict(x_test_flattened)
y_pred_labels=[np.argmax(i) for i in y_pred]
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)


plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Truth")


# In[73]:


#after adding a hiddn layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

