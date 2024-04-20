#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[6]:


data = pd.read_csv("C:\\Users\\Almas\\Downloads\\archive (2)\\dynamic_api_call_sequence_per_malware_100_0_306.csv")


# In[17]:


data.info()


# In[18]:


data.info("malware")


# In[7]:


X = data.drop(columns=['hash', 'malware'])
y = data['malware']


# In[8]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[10]:


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[11]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[12]:


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[13]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}, Loss: {loss}')


# In[19]:


from sklearn.metrics import precision_score, recall_score


# In[20]:


y_pred_probs = model.predict(X_test)


# In[21]:


y_pred = (y_pred_probs > 0.5).astype(int)


# In[22]:


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# In[23]:


print(f'Precision: {precision}')
print(f'Recall: {recall}')


# In[24]:


import matplotlib.pyplot as plt
import numpy as np


# In[25]:


y_pred_probs = model.predict(X_test)


# In[26]:


y_pred = (y_pred_probs > 0.5).astype(int)


# In[27]:


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# In[28]:


labels = ['Precision', 'Recall']
values = [precision, recall]


# In[29]:


x = np.arange(len(labels))
width = 0.35


# In[30]:


fig, ax = plt.subplots()
bars = ax.bar(x, values, width, label='Metrics')


# In[33]:


ax.set_ylabel('Scores')
ax.set_title('Precision and Recall')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


# # LSTM

# In[36]:


X = data.drop(columns=['hash', 'malware'])
y = data['malware']


# In[37]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[38]:


max_sequence_length = X.shape[1]  # Максимальная длина последовательности
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)


# In[40]:


model = Sequential([
    LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1, activation='sigmoid')
])


# In[41]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[42]:


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[43]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}, Loss: {loss}')


# In[44]:


y_pred_probs = model.predict(X_test)


# In[45]:


y_pred = (y_pred_probs > 0.5).astype(int)


# In[46]:


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# In[47]:


print(f'Precision: {precision}')
print(f'Recall: {recall}')


# In[ ]:




