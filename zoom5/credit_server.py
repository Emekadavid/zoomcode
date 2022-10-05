import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

from flask import Flask, request, jsonify


with open('model2.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
    model = pickle.load(f_in)
f_in.close()

with open('dv.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
    dv = pickle.load(f_in)
f_in.close()

def predict_single(customer, dv, model):
    X = dv.transform([customer])    
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

app = Flask("credit_server")

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    
    result = {
        'churn_probability': float(prediction)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)



