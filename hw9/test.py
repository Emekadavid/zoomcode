import requests 

data = {'url': 'https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg'}

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
#url = "https://sz0wnfbohb.execute-api.us-east-1.amazonaws.com/clothing-test/predict"

result = requests.post(url, json=data).json()
print(result)