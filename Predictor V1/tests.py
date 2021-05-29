import joblib
import pandas as pd
with open('reg.joblib', 'rb') as f:
    reg = joblib.load(f)
with open('lin.joblib', 'rb') as f:
    lin = joblib.load(f)
with open('onc.joblib', 'rb') as f:
    onc = joblib.load(f)



pred1 = reg.predict(onc.transform([['Narendra Modi Stadium',1,'Punjab Kings','Royal Challengers Bangalore',1]]))
pred2 = lin.predict(onc.transform([['Narendra Modi Stadium',1,'Punjab Kings','Royal Challengers Bangalore',1]]))
print(pred1)
print(pred2)
print((pred1 + pred2)/2)
    
