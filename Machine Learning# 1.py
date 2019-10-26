#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[5]:


data = pd.read_csv('downloads./weather/daily_weather.csv')


# In[6]:


data.columns


# In[7]:


data


# In[8]:


data[data.isnull().any(axis=1)]


# In[9]:


del data['number']


# In[10]:


before_rows = data.shape[0]
print(before_rows)


# In[11]:


data = data.dropna()


# In[12]:


after_rows = data.shape[0]
print(after_rows)


# In[13]:


before_rows - after_rows


# In[14]:


clean_data = data.copy()
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99)*1
print(clean_data['high_humidity_label'])


# In[15]:


y=clean_data[['high_humidity_label']].copy()
#y


# In[16]:


clean_data['relative_humidity_3pm'].head()


# In[17]:


y.head()


# In[18]:


morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']


# In[19]:


X = clean_data[morning_features].copy()


# In[20]:


X.columns


# In[21]:


y.columns


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)


# In[23]:


#type(X_train)
#type(X_test)
#type(y_train)
#type(y_test)
#X_train.head()
#y_train.describe()


# In[24]:


humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train, y_train)


# In[25]:


type(humidity_classifier)


# In[26]:


predictions = humidity_classifier.predict(X_test)


# In[27]:


predictions[:10]


# In[28]:


y_test['high_humidity_label'][:10]


# In[29]:


accuracy_score(y_true = y_test, y_pred = predictions)


# In[ ]:




