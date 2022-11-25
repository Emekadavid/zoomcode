import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
import numpy as np 

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x *= 1./255
    #x -= 1.
    return x

def predict(url):
    # download and resize image
    tf_image = prepare_image(download_image(url), (150,150))
    # preprocess image
    x = np.array(tf_image, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    # get the model and do inference
    interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
    interpreter.allocate_tensors()
    # get input and output index
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    # do inference now
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    # get results of prediction
    result = float(preds[0][0])

    return result 

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)

    return result    

