# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:27:16 2021

@author: Aryan
"""

def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

current_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Punjab Kings',
       'Royal Challengers Bangalore', 'Delhi Capitals',
       'Sunrisers Hyderabad']



import pandas as pd
dataset = pd.read_csv('main.csv')
dataset = dataset[(dataset['batting_team'].isin(current_teams)) &(dataset['bowling_team'].isin(current_teams))]


x = dataset.iloc[:,[0,1,2,3,5]].values
y = dataset.iloc[:,4].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import OneHotEncoder
onc = OneHotEncoder()

x_train = onc.fit_transform(x_train)
x_test = onc.transform(x_test)


from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100,max_features=None)
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
score = reg.score(x_test,y_test)*100
print("R square value:" , score)
print("Custom accuracy:" , custom_accuracy(y_test,y_pred,20))
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x_train,y_train)

y_pred = lin.predict(x_test)
score = lin.score(x_test,y_test)*100
print("R square value:" , score)


import joblib
joblib.dump(reg, 'reg.joblib')
joblib.dump(onc, 'onc.joblib')
joblib.dump(lin,'lin.joblib')
