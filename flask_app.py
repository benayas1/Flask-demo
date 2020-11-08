# import libraries
print('importing libraries...')
from pathlib import Path
from flask import Flask, request, jsonify
import logging
import random
import time
import numpy as np
import requests, os
from io import BytesIO
from src.models import *

# import settings
from settings import * # import 
print('done!\nsetting up the directories and the model structure...')

path_to_model = os.path.join(data_dir, 'models', 'model.pth')
if not os.path.exists(path_to_model):
    print('done!\nmodel weights were not found, downloading them...')
    os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
    filename = Path(path_to_model)
    r = requests.get(MODEL_URL)
    filename.write_bytes(r.content)


print('Building model and loading weights')
model = ImageClassifier(b=4)
model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
model.eval()
model.to('cpu')

print('Model loaded')

print('done!\nlaunching the server...')
# set flask params
app = Flask(__name__)

@app.route("/")
def hello():
    return "Image classification example\n"

@app.route('/predict', methods=['GET'])
def predict():
    url = request.args['url']
    app.logger.info("Classifying image %s" % (url),)
    response = requests.get(url)
    img = read(BytesIO(response.content))

    t = time.time() # get execution time
    with torch.no_grad():
        pred = model(img)
        pred = np.argmax(pred.numpy())

    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))
    app.logger.info("Image %s classified as %s" % (url, labels[pred]))

    return jsonify(labels[pred])

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)
