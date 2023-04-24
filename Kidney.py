import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('out10.csv')
# data['rbc'] = data['rbc'].replace({'normal':0,'abnormal':1})
# data['pc'] = data['pc'].replace({'normal':0,'abnormal':1})
# data['pcc'] = data['pcc'].replace({'notpresent':0,'present':1})
# data['ba'] = data['ba'].replace({'notpresent':0,'present':1})
# data['htn'] = data['htn'].replace({'yes':0,'no':1})
# data['dm'] = data['dm'].replace({'yes':0,'no':1})
# data['cad'] = data['cad'].replace({'yes':0,'no':1})
# data['appet'] = data['appet'].replace({'good':0,'poor':1})
# data['pe'] = data['pe'].replace({'yes':0,'no':1})
# data['ane'] = data['ane'].replace({'yes':0,'no':1})
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

X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)




log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)
accuracy_score(y_test,y_pred1)


log = LogisticRegression()
log.fit(X,y)





# new_data = pd.DataFrame({
#     'age':48.0,
#     'bp':80.0,
#     'sg':1.02,
#     'al':1.0,
#     'su':0.0,
#     'rbc':0.0,
#     'pc':'normal',
#     'pcc':'notpresent',
#     'ba':'notpresent',
#     'bgr':121.0,
#     'sc':36.0,
#     'sod':0.0,
#     'pot':0.0,
#     'hemo':15.4,
#     'pcv':44.0,
#     'wc':7800,
#     'rc':5.2,
#     'htn':'yes',
#     'dm':'yes',
#     'cad':'no',
#     'appet':'good',
#     'pe':'no',
#     'ane':'no',
# },index=[0])


# p = log.predict(new_data)
# if p[0] == 0:
#     print("No Diabetes")
# else:
#     print("Kidey Diseases")
