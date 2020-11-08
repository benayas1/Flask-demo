import cv2
import albumentations as A
import torch
import numpy as np

def read_img(response):
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