# import libraries
print('importing libraries...')
from pathlib import Path
from flask import Flask, request, jsonify
import logging
import random
import time
import numpy as np
import cv2
from PIL import Image
import requests, os
from io import BytesIO
import torch
import albumentations as A
from benatools.torch.efficient_net import create_efn2
from torchsummary import summary
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


class ImageClassifier(torch.nn.Module):
    def __init__(self, b=0, n_outs=39, trainable_base=False):
        super(ImageClassifier, self).__init__()
        self.base = create_efn2(b=b, include_top=False)
        
        self.set_trainable(trainable_base)
        self.classifier = torch.nn.Sequential(
          torch.nn.Linear(in_features=self.get_cnn_outputs(b), out_features=512),
          torch.nn.ReLU(),
          torch.nn.LayerNorm(512),
          torch.nn.Dropout(0.25),
          torch.nn.Linear(in_features=512, out_features=n_outs),
        )
    def set_trainable(self, trainable):
        for param in self.base.parameters():
            param.requires_grad = trainable
    def get_cnn_outputs(self, b):
        outs = [1280, 1280, 1408, 1536, 1792, 2048, 2064, 2560]
        return outs[b]
        
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x


print('Building model and loading weights')
model = ImageClassifier(b=4)
model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
model.eval()
model.to('cpu')

print('Model loaded')
def read(response):
    #img = cv2.imread(response)
    img = cv2.imdecode(np.fromstring(response.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    normalize = A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
    img = normalize(image=img)['image']
    img = np.transpose(img, axes=(2,0,1))
    #print('Image shape', img.shape)
    img = np.reshape(img, (1,3,256,256))
    img = torch.tensor(img)
    print('Image shape', img.shape)
    return img


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
        print(pred)
        pred = np.argmax(pred.numpy())
        print(pred)

    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))
    app.logger.info("Image %s classified as %s" % (url, labels[pred]))

    return jsonify(labels[pred])

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)
