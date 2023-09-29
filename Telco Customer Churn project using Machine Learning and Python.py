#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# data manupulation
# 

# In[22]:


customer_churn=pd.read_csv('C:\\Program Files\\coustmer_churn\\telico\\surya.csv')


# In[23]:


customer_churn.head()


# In[24]:


c_15=customer_churn.iloc[:,14]
c_15.head()


# In[27]:


c_random=customer_churn[(customer_churn['gender']=="Male")&(customer_churn['SeniorCitizen']==1)&(customer_churn['PaymentMethod']=="Electronic check")]


# In[29]:


c_random.head()


# In[30]:


(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)


# In[32]:


c_random=customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]


# In[33]:


c_random.head()


# In[34]:


c_random=customer_churn[(customer_churn['Contract']=="Two year") & (customer_churn['PaymentMethod']=="Mailed check")&(customer_churn['Churn']=="Yes")]


# In[35]:


c_random.head()


# In[36]:


c_333=customer_churn.sample(n=333)


# In[37]:


c_333.head()


# In[38]:


customer_churn['Churn'].value_counts()


# Data visualization
# 

# In[46]:


plt.bar(customer_churn['InternetService'].value_counts().keys().tolist(),customer_churn['InternetService'].value_counts().tolist())

plt.xlabel("Categories of Internet Service")
plt.ylabel("count")
plt.title("Distribution of Internet Service")


# In[47]:


plt.hist(customer_churn['tenure'],bins=30,color="green")


# In[48]:


plt.scatter(x=customer_churn['tenure'],y=customer_churn['MonthlyCharges'])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Monthly charges vs tenure")


# In[51]:


customer_churn.boxplot(column=['tenure'],by=['Contract'])


# Linear Regression

# In[52]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

y=customer_churn[['MonthlyCharges']]
x=customer_churn[['tenure']]


# In[53]:


y.head(),x.head()


# In[54]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


# In[55]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[56]:


regressor=LinearRegression()

regressor.fit(x_train,y_train)


# In[57]:


y_pred=regressor.predict(x_test)


# In[58]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:




