# ML_production

Example of how to deploy a trained pytorch model.

### Requeriements
Install the following packages
```
pip install torch
pip install opencv-python
pip install -U flask
pip install benatools
```

make sure port 5000 is open and run
```
python server.py
```

Open a browser and go to the following url, where <my_url> is the IP address of the server
http://<my_url>:5000/predict?url=https://people.cs.pitt.edu/~mzhang/image_ads/0/53850.jpg

