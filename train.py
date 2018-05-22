import json
import skimage
import numpy as np
import PIL
import sys
import os

sys.path.append("third_party/Mask_RCNN/")

from mrcnn import model as modellib, utils

from mrcnn.config import Config


class ActivityConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "activityobj"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    #NUM_CLASSES = 1 + 1  # Background + balloon
    NUM_CLASSES = 14

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class ActivityDataset(utils.Dataset):

    def load_activity(self, dataset_json_path):

        # Add classes:
        self.classes_names = ["Crunches mat", "Aerobic Step",
                   "Uneven bars", "Bar", "Pommel horse", "elliptical training machine", "Balance Beam",
                   "Rowing machine", "Handle", "Rope", "Bicycle", "Parallel bars", "Bars"]
                   #"head", "torso", "larm", "rarm", "lleg", "rleg"]

        for idx, c in enumerate(self.classes_names):
            self.add_class("activityobj", idx+1, c)

        with open(dataset_json_path) as dataset_json_file:
            self.json_data = json.load(dataset_json_file)

        print("Elements in the json file:", str(len(self.json_data)))

        for image_path, masks in self.json_data.items():
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image("activityobj",
                           image_id=image_path,  # use file path as a unique image id
                           path=image_path,
                           width=width, height=height)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(self.json_data[info["id"]])], dtype=np.uint8)
        lbls = np.zeros(len(self.json_data[info["id"]]), dtype=np.int32)

        for idx, (mask_path, mask_info) in enumerate(self.json_data[info["id"]].items()):
            mask_class = mask_info["class"]
            mask[:,:,idx] = np.array(PIL.Image.open(mask_path), dtype=np.uint8)
            lbls[idx] = self.classes_names.index(mask_class) + 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), lbls

    #def image_reference(self, image_id):
    #    """Return the path of the image."""
    #    info = self.image_info[image_id]
    #    if info["source"] == "balloon":
    #        return info["path"]
    #    else:
    #        super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ActivityDataset()
    dataset_train.load_activity("dataset/trainval/train.json")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ActivityDataset()
    dataset_val.load_activity("dataset/trainval/val.json")
    dataset_val.prepare()
    #dataset_val = dataset_train

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

if __name__ == '__main__':

    config = ActivityConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir="logs/")

    weights_path = "weights/mask_rcnn_coco.h5"
    # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)

    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    train(model)
