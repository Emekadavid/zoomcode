import pickle
from sklearn.linear_model import LogisticRegression

with open('model1.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
    model = pickle.load(f_in)
f_in.close()

with open('dv.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
    dv = pickle.load(f_in)
f_in.close()

customer = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X_test = dv.transform(customer)

prediction = model.predict_proba(X_test)[:,1]
print(prediction)


