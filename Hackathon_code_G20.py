#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[21]:


df=pd.read_csv("Training Dataset.csv")
df2=pd.read_csv("Evaluation Dataset.csv")


# In[22]:


X=pd.DataFrame(df.iloc[:,0:33])
Y=pd.DataFrame(df.iloc[:,-1])


# In[23]:


Y


# In[26]:


df.describe()


# In[43]:





# In[38]:





# In[27]:


model=linear_model.LinearRegression()


# In[28]:


model.fit(X, Y)


# In[ ]:





# In[30]:


X2=pd.DataFrame(df2.iloc[:,:])


# In[31]:


X2.describe()


# In[34]:


Y2=model.predict(X2)


# In[35]:


Yp = pd.DataFrame(Y2, columns=['Predicted'])
Yp.to_csv("predicted2.csv")


# In[ ]:




