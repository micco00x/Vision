import json
import skimage
import skimage.draw
import skimage.io
import numpy as np
import urllib.request
import os
import scipy.misc



image_path = "image.png"

json_filename = "rowingMachine.json"
with open(json_filename) as json_file:
    json_data = json.load(json_file)

elem = json_data[0]
img_url = elem["Labeled Data"]
if not os.path.isfile(image_path):
    urllib.request.urlretrieve(img_url, image_path)
image = skimage.io.imread(image_path)
h, w = image.shape[:2]

poly_coord = elem["Label"]["Rowing machine"][0]
x = [coord["x"] for coord in poly_coord]
y = [coord["y"] for coord in poly_coord]

rr, cc = skimage.draw.polygon(y, x, shape=(h, w))
mask = np.zeros([h, w])
mask[rr, cc] = 1

mask = np.flipud(mask)

scipy.misc.imsave("mask.png", mask)
