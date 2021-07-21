#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


data = pd.read_csv(r'E:/ML Notes/Melbourne_housing_FULL.csv')


# In[66]:


data.head()


# In[67]:


data.corrwith(data['Price'])


# In[68]:


data.shape[0]


# In[69]:


data.duplicated().sum()


# In[70]:


data.drop_duplicates(keep=False, inplace=True)


# In[71]:


data.shape[0]


# In[72]:


data.isnull().sum()


# In[73]:


fig, ax = plt.subplots(figsize=(15,7))
sns.heatmap(data.isnull(), yticklabels=False, cmap='viridis')


# In[74]:


data.dropna(subset=['CouncilArea','Lattitude','Longtitude','Regionname','YearBuilt'], inplace=True)


# In[75]:


data.isnull().sum()


# In[76]:


data['Price'].fillna(data['Price'].median(), inplace= True)
data['Bathroom'].fillna(data['Bathroom'].value_counts().index[0], inplace= True)
data['Car'].fillna(data['Car'].value_counts().index[0], inplace= True)
data['Landsize'].fillna(data['Landsize'].mean(), inplace= True)
data['BuildingArea'].fillna(data['BuildingArea'].mean(), inplace= True)


# In[77]:


data.isnull().sum()


# In[78]:


data.dtypes


# In[79]:


df = data.select_dtypes(['object']).columns


# In[80]:


data[df] = data[df].astype('category')


# In[81]:


data.dtypes


# In[82]:


data['Date'] = pd.to_datetime(data['Date'])


# In[83]:


data['Postcode'] = data['Postcode'].astype('category')


# In[84]:


data['Bedroom2'] = data['Bedroom2'].astype('int')


# In[85]:


data['Bathroom'] = data['Bathroom'].astype('int')
data['Car'] = data['Car'].astype('int')


# In[86]:


data.info()


# In[24]:


data.describe()


# In[87]:


data['SellerG'].value_counts().head(5)


# In[88]:


plt.subplots(figsize=(15,11))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')


# In[89]:


data.info()


# In[90]:


data.drop(['Date','Suburb','SellerG','Address','Postcode'], axis=1, inplace=True)


# In[91]:


data.dtypes


# In[92]:


x = data.iloc[:,:-1].values


# In[93]:


y = data.iloc[:,-1].values


# In[94]:


res = pd.get_dummies(data, columns=['Type','Method','CouncilArea','Regionname'], drop_first=True)


# In[95]:


x = res


# In[96]:


from sklearn.model_selection import train_test_split


# In[97]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,
                                                random_state=0)


# In[98]:


from sklearn.linear_model import LinearRegression


# In[99]:


mr = LinearRegression()


# In[100]:


mr.fit(x_train,y_train)


# In[101]:


y_pred = mr.predict(x_test)


# In[102]:


y_pred


# In[103]:


from sklearn.metrics import r2_score


# In[104]:


r2 = r2_score(y_pred,y_test)*100


# In[105]:


r2


# In[106]:


adj_r2score = ((1-r2)/(15539/15475))


# In[107]:


adj_r2score


# In[ ]:




