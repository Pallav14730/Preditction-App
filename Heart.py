import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('heart-disease.csv')
print(data)
data.isnull().sum()
data_dup = data.duplicated().any()
data = data.drop_duplicates()

cat_val = []
num_val = []

for column in data.columns:
    if data[column].nunique() < 10:
        cat_val.append(column)

    else:
        num_val.append(column)
print(cat_val)
print(num_val)

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)
accuracy_score(y_test,y_pred1)


log = LogisticRegression()
log.fit(X,y)
print(data)
new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
    'slope':2,
    'ca':2,
    'thal':3,
},index=[0])


p = log.predict(new_data)
if p[0] == 0:
    print("No Diseases")
else:
    print("Heart Diseases")

# print(X_train)
# print(data)