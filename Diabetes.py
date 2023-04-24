from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv('diabetes.csv')
data.isnull().sum()
data_dup = data.duplicated().any()
data = data.drop_duplicates()

# how to freeze a model

cat_val = []
num_val = []

for column in data.columns:
    if data[column].nunique() < 10:
        cat_val.append(column)

    else:
        num_val.append(column)

X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)
accuracy_score(y_test,y_pred1)


log = LogisticRegression()
log.fit(X,y)

# vfnvjf
def request():
    new_data = pd.DataFrame({
        'Pregnancies':6,
        'Glucose':85,
        'BloodPressure':66,
        'SkinThickness':29,
        'Insulin':0,
        'BMI':26.6,
        'DiabetesPedigreeFunction':0.351,
        'Age':31,
    },index=[0])


p = log.predict(new_data1)
if p[0] == 0:
    print("No Diabetes")
else:
    print("Diabetes")
