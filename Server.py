from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



app = Flask(__name__)

@app.route('/')
def navbar():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("About.html")


@app.route('/predictions')
def prediction():
    return render_template("Prediction.html")

@app.route('/result2', methods=['POST'])
def result2():
    data = pd.read_csv('heart-disease.csv')
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

    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']
    
    
    new_data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
    }, index=[0])



    p = log.predict(new_data)
    if p[0] == 0:
        age = age;
        sex = sex;
        cp = cp;
        trestbps = trestbps;
        chol = chol
        fbs = fbs
        restecg = restecg
        thalach = thalach
        exang = exang
        oldpeak = oldpeak
        slope = slope
        ca = ca
        thal = thal
        return render_template('result4.html',age=age,sex=sex,cp=cp,trestbps=trestbps,chol=chol,fbs=fbs,restecg=restecg,thalach=thalach,exang=exang,oldpeak=oldpeak,slope=slope,ca=ca,thal=thal)
    


    else:
        age = age;
        sex = sex;
        cp = cp;
        trestbps = trestbps;
        chol = chol
        fbs = fbs
        restecg = restecg
        thalach = thalach
        exang = exang
        oldpeak = oldpeak
        slope = slope
        ca = ca
        thal = thal
        return render_template('result3.html',age=age,sex=sex,cp=cp,trestbps=trestbps,chol=chol,fbs=fbs,restecg=restecg,thalach=thalach,exang=exang,oldpeak=oldpeak,slope=slope,ca=ca,thal=thal)
    


@app.route('/result', methods=['POST'])
def result():
    
    data = pd.read_csv('diabetes.csv')
    
    
    print("Hello world")
   

    data.isnull().sum().sum()
    data_dup = data.duplicated().any()
    data = data.drop_duplicates()

    cat_val = []
    num_val = []

    for column in data.columns:
        if data[column].nunique() < 10:
            cat_val.append(column)

        else:
            num_val.append(column)

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log = LogisticRegression()
    log.fit(X_train, y_train)
    y_pred1 = log.predict(X_test)
    accuracy_score(y_test, y_pred1)

    log = LogisticRegression()
    log.fit(X, y)

    
    Pregnancies = request.form['Pregnancies']
    Glucose = request.form['Glucose']
    BloodPressure = request.form['BloodPressure']
    SkinThickness = request.form['SkinThickness']
    Insulin = request.form['Insulin']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    Age = request.form['Age']

   
    
    new_data = pd.DataFrame({
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age,
    }, index=[0])

    
    p = log.predict(new_data)
    if p[0] == 0:
        age1 = Age;
        glucose1 = Glucose;
        bp1 = BloodPressure;
        st1 = SkinThickness;
        in1 = Insulin;
        bmi1 = BMI;
        dp1 = DiabetesPedigreeFunction;
        preg1 = Pregnancies;
        return render_template('result2.html',age=age1,glucose=glucose1,bp=bp1,st=st1,in1=in1,bmi=bmi1,dp=dp1,preg=preg1)
        

    else:
        age1 = Age;
        glucose1 = Glucose;
        bp1 = BloodPressure;
        st1 = SkinThickness;
        in1 = Insulin;
        bmi1 = BMI;
        dp1 = DiabetesPedigreeFunction;
        preg1 = Pregnancies;
        return render_template('result.html',age=age1,glucose=glucose1,bp=bp1,st=st1,in1=in1,bmi=bmi1,dp=dp1,preg=preg1)
    
      
    
    

@app.route('/result5', methods=['POST'])
def result5():

    data = pd.read_csv('mac123.csv')
   

    
    data.isnull().sum().sum()
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log = LogisticRegression()
    log.fit(X_train, y_train)
    y_pred1 = log.predict(X_test)
    accuracy_score(y_test, y_pred1)

    log = LogisticRegression()
    log.fit(X, y)


    age = request.form['age']
    bp = request.form['bp']
    sg = request.form['sg']
    al = request.form['al']
    su = request.form['su']
    rbc = request.form['rbc']
    pc = request.form['pc']
    pcc = request.form['pcc']
    ba = request.form['ba']
    bgr = request.form['bgr']
    bu = request.form['bu']
    sc = request.form['sc']
    sod = request.form['sod']
    pot = request.form['pot']
    hemo = request.form['hemo']
    pcv = request.form['pcv']
    wc = request.form['wc']
    rc = request.form['rc']
    htn = request.form['htn']
    dm = request.form['dm']
    cad = request.form['cad']
    appet = request.form['appet']
    pe = request.form['pe']
    ane = request.form['ane']
    


    
    new_data = pd.DataFrame({
        'age': age,
        'bp': bp,
        'sg': sg,
        'al': al,
        'su': su,
        'rbc': rbc,
        'pc': pc,
        'pcc': pcc,
        'ba': ba,
        'bgr': bgr,
        'bu': bu,
        'sc': sc,
        'sod': sod,
        'pot': pot,
        'hemo': hemo,
        'pcv': pcv,
        'wc': wc,
        'rc': rc,
        'htn': htn, 
        'dm': dm,
        'cad': cad,
        'appet': appet,
        'pe': pe,
        'ane': ane,
       
    }, index=[0])
    
    p = log.predict(new_data)
    if p[0] == 0:
    #    return f"<h1> Not Suffered </h1>"
        bp1 = bp 
        sg1 = sg
        al1 = al
        su1 = su
        rbc1 = rbc
        pc1 = pc
        pcc1 = pcc
        ba1 =ba
        bgr1 = bgr
        bu1 = bu
        sc1 = sc
        sod1 = sod
        pot1 = pot
        hemo1 = hemo
        pcv1 = pcv
        wc1 = wc
        rc1 = rc
        htn1 = htn
        dm1 = dm
        cad1 = cad
        appet1 = appet
        pe1 = pe
        ane1 = ane
        return render_template('result6.html',bp=bp1,sg=sg1,al=al1,su=su1,rbc=rbc1,pc=pc1,pcc=pcc1,ba=ba1,bgr=bgr1,bu=bu1,sc=sc1,sod=sod1,pot=pot1,hemo=hemo1,pcv=pcv1,wc=wc1,rc=rc1,htn=htn1,dm=dm1,cad=cad1,appet=appet1,pe=pe1,ane=ane1)
        
     

    else:
        # return f"<h1> Suffered </h1>"
        bp1 = bp 
        sg1 = sg
        al1 = al
        su1 = su
        rbc1 = rbc
        pc1 = pc
        pcc1 = pcc
        ba1 =ba
        bgr1 = bgr
        bu1 = bu
        sc1 = sc
        sod1 = sod
        pot1 = pot
        hemo1 = hemo
        pcv1 = pcv
        wc1 = wc
        rc1 = rc
        htn1 = htn
        dm1 = dm
        cad1 = cad
        appet1 = appet
        pe1 = pe
        ane1 = ane
        return render_template('result5.html',bp=bp1,
        sg=sg1,al=al1,su=su1,rbc=rbc1,pc=pc1,
        pcc=pcc1,ba=ba1,bgr=bgr1,bu=bu1,sc=sc1,
        sod=sod1,pot=pot1,hemo=hemo1,pcv=pcv1,wc=wc1,
        rc=rc1,htn=htn1,dm=dm1,cad=cad1,appet=appet1,
        pe=pe1,ane=ane1)
        
      
@app.route('/Kidney')
def Kidney():
    return render_template('Kidney.html')


@app.route('/Diabetes')
def Diabetes():
    return render_template('Diabetes.html')


@app.route('/Heart')
def Heart():
    
    
    
    
    return render_template("Heart.html")


@app.route('/HeartDisease', methods=['POST'])
def HeartDisease():
    return render_template('HeartDisease.html')

@app.route('/about-heart-disease')
def aboutheartdisease():
    return render_template('About-Heart-Disease.html')

@app.route('/about-kidney-disease')
def aboutkidneydisease():
    return render_template('About-Kidney-Disease.html')

@app.route('/about-diabetes')
def aboutdiabetes():
    return render_template('About-Diabetes.html')


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=8080)
