import json
import PIL.Image as Image
from PIL import ImageDraw, ImageFont
import MaskExam as Mask
import numpy as np
import math
import os
import common

path = "dataset/trainval/"
jsonName = "val.json"

ret1 = []
ret2 = []

jsonPath = path + "/" + jsonName
b = json.load(open(jsonPath))

immNum = 0
vector = []

iter = 0

idx = 0
tot = len(b)
idx_2 = 0
success = 0
total = 0
classes = {}


def coordToMatrix(coord, w, h):
	img_size = (w, h)
	poly = Image.new("RGB", img_size)
	pdraw = ImageDraw.Draw(poly)
	pdraw.polygon(coord,
				  fill=(255,255,255), outline=(255,255,255))
	#poly = poly.transpose(Image.FLIP_LEFT_RIGHT)
	#poly = poly.rotate(180)
	#pix = np.array(poly.getdata()).reshape(w, h)
	return poly


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

def find_max_coord(x, y):
	x_max = 0
	x_min = 10000000
	y_max = 0
	y_min = 10000000
	for indice in range(len(x)):
		if x[indice] < x_min:
			x_min = x[indice]
		if y[indice] < y_min:
			y_min = y[indice]
		if x[indice] > x_max:
			x_max = x[indice]
		if y[indice] > y_max:
			y_max = y[indice]
			
	return [x_max, x_min, y_max, y_min]

def cade_internamente(max, centroide):
	if centroide[0]< max[0] and centroide[0] > max[1]:
		if centroide[1]< max[2] and centroide[1] > max[3]:
			return True
	return False

iter= 0

for json_elem in b:

	# read image for sizes
	iter+= 1
	try:
		im = Image.open(json_elem).convert('RGB')
	except:
		print("%d/%d" % (iter,tot), "[MISSING]", json_elem)
		continue
	
	print("%d/%d" % (iter,tot), "[CHECKING]", json_elem)
	
	w, h = im.size
	
	im = np.array(im)
	seg = b[json_elem]
	for m in seg:
		mask = seg[m]
		name = seg[m]['class']

		maskMat = []
		idClassi = []

		idss = []
		centroidi_lista = []
		aree = []
		max_coord = []
		#for i in seg.keys():
		#	name = str(i)
		class_id_name = common.all_classes.index(name)

		idClassi.append(classes.get(name))
		x_coord = []
		y_coord = []
		for k in range(len(seg[m]['polygon'][0]['all_points_x'])):
			y_coord.append(seg[m]['polygon'][0]['all_points_y'][k])
			x_coord.append(seg[m]['polygon'][0]['all_points_x'][k])
		coord = []
		for ind in range(len(x_coord)):
			coord.append(x_coord[ind])
			coord.append(y_coord[ind])
		immagine = coordToMatrix(coord, w, h)
		
		# not used. In any case can generate an exception because of mistakes in json file (mask coordinate)
		#centroidi_lista.append(find_centroid(immagine))
		aree.append(compute_area(immagine))
		idss.append(class_id_name)
		max_coord.append(find_max_coord(x_coord, y_coord))

	centroidi_lista_mask, idss_mask, aree_mask = Mask.centreAnalisi(im, w, h)
	for indice in range(len(idss)):
		total += 1
		for indice_mask in range(len(idss_mask)):
			# considering only the network areas in a range of 50%-150% of the mask in the original image
			if (aree[indice] * 0.5) < aree_mask[indice_mask] and aree_mask[indice_mask] < (aree[indice] * 1.5):
				# check if network centroids fall inside the bounding box of in the  the mask original image
				if cade_internamente(max_coord[indice], centroidi_lista_mask[indice_mask]):
					if idss_mask[indice_mask] == idss[indice]:
						success += 1



print("Numero di successi: " + str(success))
print("Numero totale label: " + str(total))
print("Percentuale di successo: "+ str(float(success) / float(total) ) + "%")
