from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model=joblib.load("best_credit_model.pkl")


class CreditData(BaseModel):
    Age:int
    Gender:str
    Martial_Status:str
    Education_Level:str
    Employment_Status:str
    Credit_Utilization_Ratio:float
    Payment_History:float
    Number_of_Credit_Accounts:int
    Loan_Amount:float
    Interest_Rate:float
    Loan_Term:int
    Type_of_Loan:str

app=FastAPI()
@app.get('/')
def home():
    return{"message":"Welcome to Credit Scoring API"}

@app.post('/predict')
def predict_credit_risk(data:CreditData):
    data=data.dict()
    print(data)
    input_data=np.array([[data['Age'],data['Gender'],data['Martial_Status'],data['Education_Level'],
                          data['Employment_Status'],data['Credit_Utilization_Ratio'],data['Payment_History'],
                          data['Number_of_Credit_Accounts'],
                          data['Loan_Amount'],data['Interest_Rate'],data['Loan_Term'],data['Type_of_Loan']]])
    prediction=model.predict(input_data)
    print(prediction)



def home():
    return{"message":"Welcome to Credit Scoring Model API"}