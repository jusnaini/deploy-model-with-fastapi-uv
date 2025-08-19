#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for an updated module 5 of ML Zoomcamp
# 
# The code is based on the modules 3 and 4. We use the same dataset: [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# In[4]:

import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pickle

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[]: load_data

def load_data():
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)
    return df


# In[]: train_model 

def train_model(df):
    numerical = ['tenure', 'monthlycharges', 'totalcharges']
    categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
    ]

    pipeline = make_pipeline(DictVectorizer(),LogisticRegression(solver='liblinear'))
    train_dict = df[categorical + numerical].to_dict(orient='records')
    y_train = df.churn
    pipeline.fit(train_dict, y_train)
    return pipeline




# In[]:save_model
def save_model(filename,model):
    # save the model
    with open (filename,'wb') as f_out:
        pickle.dump(model,f_out)
    print (f'Model saved to {filename}')

# In[] Main
df = load_data()

pipeline = train_model(df)

save_model('model.bin',pipeline)



# In[13]:


# customer = {
#     'gender': 'male',
#      'seniorcitizen': 0,
#      'partner': 'yes',
#      'dependents': 'yes',
#      'phoneservice': 'no',
#      'multiplelines': 'no_phone_service',
#      'internetservice': 'dsl',
#      'onlinesecurity': 'no',
#      'onlinebackup': 'yes',
#      'deviceprotection': 'no',
#      'techsupport': 'no',
#      'streamingtv': 'no',
#      'streamingmovies': 'no',
#      'contract': 'month-to-month',
#      'paperlessbilling': 'yes',
#      'paymentmethod': 'electronic_check',
#      'tenure': 6,
#      'monthlycharges': 29.85,
#      'totalcharges': 29.85,
# }


# # In[14]:


# X = dv.transform(customer)


# # In[15]:


# churn = model.predict_proba(X)[0,1]





# # ## Saving model

# # In[18]:


# import pickle


# # In[19]:


# with open ('model.bin','wb') as f_out:
#     pickle.dump((dv,model),f_out)


# # ## For loading the model

# # In[22]:


# with open('model.bin','rb') as f_in:
#     (dv_in,model_in)=pickle.load(f_in)


# # In[23]:


# # Example

# # customer = {
# #     'gender': 'female',
# #      'seniorcitizen': 0,
# #      'partner': 'yes',
# #      'dependents': 'yes',
# #      'phoneservice': 'no',
# #      'multiplelines': 'no_phone_service',
# #      'internetservice': 'dsl',
# #      'onlinesecurity': 'no',
# #      'onlinebackup': 'yes',
# #      'deviceprotection': 'no',
# #      'techsupport': 'no',
# #      'streamingtv': 'no',
# #      'streamingmovies': 'no',
# #      'contract': 'month-to-month',
# #      'paperlessbilling': 'yes',
# #      'paymentmethod': 'electronic_check',
# #      'tenure': 8,
# #      'monthlycharges': 100,
# #      'totalcharges': 105,
# # }

# # X = dv_in.transform(customer)
# # churn = model_in.predict_proba(X)[0,1]

# # if churn >= 0.5:
# #     print('Send email with promo!')
# # else:
# #     print ('Dont do anything!')


# # ## Pipeline
# # 
# # Instead of saving model in the form of tuple (dv,model), more convenient to implement both into pipeline. So that only need to 
# # call single object to do dict vectorization followed by model fit: 
# # 

# # In[25]:




# # In[26]:




# # In[27]:


# # dv = DictVectorizer()

# # train_dict = df[categorical + numerical].to_dict(orient='records')
# # X_train = dv.fit_transform(train_dict)

# # model = LogisticRegression(solver='liblinear')
# # model.fit(X_train, y_train)





# # In[33]:


# ## usage
# # customer

# # churn = pipeline.predict_proba(customer)[0,1]
# # print(f'prob of churning = {churn*100:.2f}%')
# # if churn >= 0.5:
# #     print('Send email with promo!')
# # else:
# #     print ('Dont do anything!')





# # In[ ]:


# # # load model
# # with open ('model.bin','rb') as f_in:
# #     pipeline = pickle.load(f_in)

