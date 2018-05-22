import json
import urllib.request
import os

import skimage.draw
import skimage.io

import numpy as np
import scipy.misc

USE_PASCAL = False

json_output_filename = "dataset.json"
dataset_labelbox_path = "dataset/labelbox/"
activities = ["balanceBeam", "doingCrunches", "ellipticaltrainer", "parallelBars", "pommelHorse", "rowingMachine",
              "spinningHR", "stepAerobics", "unevenBars"]

pascal_images_dir = "dataset/Pascal-Part Dataset/trainval/examples/"
pascal_masks_dir = "dataset/Pascal-Part Dataset/trainval/masks/"

dataset_path = "dataset/trainval/"
dataset = {}

# Create dataset directory if it doesn't exist:
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

if not os.path.exists(dataset_path + "images/"):
    os.makedirs(dataset_path + "images/")

if not os.path.exists(dataset_path + "masks/"):
    os.makedirs(dataset_path + "masks/")

# Read all json files taken from LabelBox:
for activity in activities:
    json_path = dataset_labelbox_path + activity + ".json"
    print("Reading", json_path)
    skipped = 0
    # Open json file:
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    # Extract data from json file:
    for idx, elem in enumerate(json_data):
        if elem["Label"] == "Skip":
            continue

        masks_dict = {}

        # Download image:
        img_url = elem["Labeled Data"]
        img_path = dataset_path + "images/" + activity + "_" + str(idx) + ".png"
        if not os.path.isfile(img_path):
            try:
                urllib.request.urlretrieve(img_url, img_path)
            except TimeoutError:
                print("TimeoutError")
                continue
        image = skimage.io.imread(img_path)
        h, w = image.shape[:2]

        # Create masks from the polygons:
        mask_idx = 0
        for mask_class, mask_class_coordinates in elem["Label"].items():
            if mask_class in ["Doing crunches hands", "Leg", "Floor"]:
                continue
            for poly_coord in mask_class_coordinates:
                #print("poly_coord:", poly_coord)

                if poly_coord == "blur" or poly_coord == "pixelated":
                    skipped = skipped + 1
                    continue

                x = [coord["x"] for coord in poly_coord]
                y = [coord["y"] for coord in poly_coord]

                polygon = [{'all_points_y': y, 'name': 'polygon', 'all_points_x': x}]

                mask_path = dataset_path + "masks/" + activity + "_" + str(idx) + "_" + str(mask_idx) + ".png"

                if not os.path.isfile(mask_path):
                    rr, cc = skimage.draw.polygon(y, x, shape=(h, w))
                    mask = np.zeros([h, w])
                    mask[rr, cc] = 1
                    mask = np.flipud(mask)
                    scipy.misc.imsave(mask_path, mask)

                masks_dict[mask_path] = {"class": mask_class, "polygon": polygon}
                mask_idx = mask_idx + 1

        # Download masks:
        #if not "Masks" in elem.keys():
        #    print("Skipping the image, Masks not in keys.")
        #    skipped = skipped + 1
        #else:
        #    # Download image from URL:
        #    for mask_idx, (mask_class, mask_url) in enumerate(elem["Masks"].items()):
        #        if mask_url == "error":
        #            continue
        #        mask_path = dataset_path + "masks/" + activity + "_" + str(idx) + "_" + str(mask_idx) + ".png"
        #        if not os.path.isfile(mask_path):
        #            try:
        #                urllib.request.urlretrieve(mask_url, mask_path)
        #            except TimeoutError:
        #                print("TimeoutError")
        #                continue
        #        masks_dict[mask_path] = mask_class

        if masks_dict != {}:
            dataset[img_path] = masks_dict

    print("Total skipped images for " + json_path + ": " + str(skipped))

if USE_PASCAL:
    # Read images of Pascal-Part dataset and related masks (those which start with the same name):
    pascal_images_filenames = os.listdir(pascal_images_dir)
    pascal_masks_filenames = os.listdir(pascal_masks_dir)
    pascal_images_filenames.sort()
    pascal_masks_filenames.sort()

    idx_mask = 0
    for idx_image, image_filename in enumerate(pascal_images_filenames):
        masks_dict = {}
        while pascal_masks_filenames[idx_mask].startswith(image_filename[:image_filename.rfind(".")]):
            mask_filename = pascal_masks_filenames[idx_mask]
            mask_class = mask_filename[mask_filename.rfind("_")+1:mask_filename.rfind(".")]
            mask_path = pascal_masks_dir + mask_filename
            masks_dict[mask_path] = {"class": mask_class}
            idx_mask = idx_mask + 1
            if idx_mask >= len(pascal_masks_filenames):
                break
        if masks_dict != {}:
            image_path = pascal_images_dir + image_filename
            dataset[image_path] = masks_dict
        if idx_mask >= len(pascal_masks_filenames):
            break

# Write dictionary onto a JSON file:
with open(dataset_path + json_output_filename, "w") as fp:
    json.dump(dataset, fp)
