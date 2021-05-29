# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:54:23 2021

@author: Aryan
"""


def predictRuns(testInput):
    import joblib
    import pandas as pd
    
    with open('reg.joblib', 'rb') as f:
        reg = joblib.load(f)
    with open('lin.joblib', 'rb') as f:
        lin = joblib.load(f)
    with open('onc.joblib', 'rb') as f:
        onc = joblib.load(f)
    
    i = pd.read_csv(testInput)
    
    
    
    z =  i["batsmen"].values
    
    wickets = -1;
    z = str(z)
    for j in z:
        if j == ',':
            wickets += 1;
    
    input_list = [[i['venue'].values[0] ,i['innings'].values[0], i['batting_team'].values[0], i['bowling_team'].values[0], wickets]]

    pred1 = reg.predict(onc.transform(input_list))
    pred2 = lin.predict(onc.transform(input_list))
    prediction = (pred1 + pred2)/2
    

    prediction = round(int(prediction))

        
    return prediction