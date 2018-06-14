import os
import sys
sys.path.append("third_party/Mask_RCNN/")
sys.path.append('cocoapi/PythonAPI')
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
#from samples.coco import coco
import activity
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import cv2
from PIL import Image
import common

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained
MODEL_DIR = os.path.join(ROOT_DIR, "logs/")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco_0080.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
   utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join("images/")

class InferenceConfig(activity.ExtendedCocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	DETECTION_MIN_CONFIDENCE = 0.6
	a = common.activity_classes_names + common.coco_classes
	NUM_CLASSES = 1+len(a)

config = InferenceConfig()
# config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ["background"] + common.coco_classes + common.activity_classes_names

def get_ax(rows=1, cols=1, size=16):
	_, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
	return ax

def find_person(fig):
	indexList = [0,1,2,3,4,5,6,7,8]
	results = model.detect([fig], verbose=1)
	r = results[0]
	ax = get_ax(1)
	# ['BG', 'screwdriver', 'belt', 'guard', 'mesh', 'spanner', 'boh1', 'boh2'], r['scores']
	visualize.display_instances(fig, r['rois'], r['masks'], r['class_ids'], class_names , r['scores'], ax=ax, title="Predictions")
	plt.show()
	ids = r['class_ids']
	ret = []
	for i in indexList:
		ret.append(np.count_nonzero(ids == i))
	return ids

	
def hex_to_rgb(value):
	value = value.lstrip('#')
	return tuple(int(value[i:i+2], 16)/255.0 for i in (0, 2 ,4))

	
def generate_class_colors():
	colors = []
	for hexcolor in common.hex_colors:
		colors.append(hex_to_rgb(hexcolor))
	return colors
	

def process_masked_image(image, boxes, masks, class_ids, class_names, scores=None,
					  show_mask=True, show_bbox=True, colors=None, captions=None):
	"""
	boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
	masks: [height, width, num_instances]
	class_ids: [num_instances]
	class_names: list of class names of the dataset
	scores: (optional) confidence scores for each box
	show_mask, show_bbox: To show masks and bounding boxes or not
	colors: (optional) An array or colors to use with each object
	captions: (optional) A list of strings to use as captions for each object
	"""
	# Number of instances
	N = boxes.shape[0]
	if N:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
	
	# Generate random colors
	dcolors = colors or visualize.random_colors(N)
	colors = generate_class_colors()
	# Show area outside image boundaries.
	height, width = image.shape[:2]

	font = cv2.FONT_HERSHEY_COMPLEX_SMALL

	masked_image = image.astype(np.uint8).copy()
	for i in range(N):
		
		color = dcolors[i]
		
		y1, x1, y2, x2 = boxes[i]
		# Label
		if not captions:
			class_id = class_ids[i]
			score = scores[i] if scores is not None else None
			label = class_names[class_id]
			x = random.randint(x1, (x1 + x2) // 2)
			caption = "{} {:.3f}".format(label, score) if score else label
			color = colors[class_id]
		else:
			caption = captions[i]
			

		
		# Mask
		mask = masks[:, :, i]
		masked_image = visualize.apply_mask(masked_image, mask, color)
		
		#modify only after the mask application
		color = tuple([int(ch*255) for ch in color])
		
		# Bounding box
		if not np.any(boxes[i]):
			# Skip this instance. Has no bbox. Likely lost in image cropping.
			continue
		
		cv2.rectangle(masked_image, (x1,y1), (x2,y2), color, 2)


		cv2.putText(masked_image, caption, (x1,y1+8), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

	return masked_image	

	
def apply_masks(fig):
	results = model.detect([fig], verbose=0)
	r = results[0]
	masked_image = process_masked_image(fig, r['rois'], r['masks'], r['class_ids'], class_names , r['scores'])
	return masked_image



def find_centroid(im):
	width, height = im.size
	
	XX, YY, count = 0, 0, 0
	for x in range(0, width, 1):
		for y in range(0, height, 1):
			if im.getpixel((x, y)) == (255,255,255):
				XX += x
				YY += y
				count += 1
	return XX/count, YY/count

def compute_area(im):
	width, height = im.size
	area = 0
	for x in range(0, width, 1):
		for y in range(0, height, 1):
			if im.getpixel((x, y)) == (255,255,255):
				area += 1
	return area

def centreAnalisi(fig, w, h):

	results = model.detect([fig], verbose=0)
	r = results[0]
	ids = r['class_ids']
	maschere = r["masks"]
	
	numMask = 0
	try:
		numMasks = len(maschere[0][0])
	except Exception as e:
		print(e)
		return 0

	maskRet = []
	for i in range(numMasks):
		img = np.zeros([h, w, 3], dtype=np.uint8)
		maskRet.append(img)
	for hh in range(maschere.shape[0]):
		for ww in range(maschere.shape[1]):
			for mm in range(maschere.shape[2]):
				if maschere[hh][ww][mm]: # mask from network are height*width*different mask (bool)
					maskRet[mm][hh][ww] = (255,255,255) #im pil store in height width

	'''
	for c in range(3):
	img[:, :, c] = np.where(indice == 1, 255, img[:, :, c])
	'''
	centroidi_ret= []
	aree = []
	for maskSingle in maskRet:
		image = Image.fromarray(maskSingle, 'RGB')
		#ww, hh = image.size
		aree.append(compute_area(image))
		ret = find_centroid(image)
		centroidi_ret.append(ret)
	return centroidi_ret, ids, aree

def test():
	file_names = next(os.walk(IMAGE_DIR))[2]
	for f in file_names:
		image = skimage.io.imread(os.path.join(IMAGE_DIR, f))
		number = find_person(image)
		print(number)

if __name__ == "__main__":
	test()