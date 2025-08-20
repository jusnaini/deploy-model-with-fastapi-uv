# marketing.py
import requests

url = 'http://localhost:9696/predict'

customer = {
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "yes",
  "dependents": "yes",
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "tenure": 8,
  "monthlycharges": 100,
  "totalcharges": 105
}

churn = requests.post(url, json=customer).json()['churn_probability']

print(f'prob of churning = {churn*100:.2f}%')
if churn >= 0.5:
    print('Send email with promo!')
else:
    print ('Dont do anything!')