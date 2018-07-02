import os
import sys
import shutil
import random
# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from MRIMathConfig import MRIMathConfig
from FlairDataset import FlairDataset


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# stuff to run always here such as class/def
def main():
    config = MRIMathConfig()
    config.display()
    
    random.seed(12345)
    data_dir = "Data/BRATS_2018/HGG"
    val_dir = "Data/BRATS_2018/HGG_Validation"
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    test_dir = "Data/BRATS_2018/HGG_Testing"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    list_imgs = os.listdir(data_dir)

    if os.listdir(val_dir) == []:
        # split validation data (20% of dataset)
        val_imgs = random.sample(list_imgs, round(0.1*len(list_imgs)))
        for sub_dir in list_imgs:
            if sub_dir in val_imgs:
                dir_to_move = os.path.join(data_dir, sub_dir)
                shutil.move(dir_to_move, val_dir)
                list_imgs.remove(sub_dir)
                
    if os.listdir(test_dir) == []:
        # split testing data (5% of dataset)
        test_imgs = random.sample(list_imgs, round(0.05*len(list_imgs)))
        for sub_dir in list_imgs:
            if sub_dir in test_imgs:
                dir_to_move = os.path.join(data_dir, sub_dir)
                shutil.move(dir_to_move, test_dir)
    """
    list_imgs = os.listdir(aug_dir)
    aug_imgs = random.sample(list_imgs, round(0.5*len(list_imgs)))
    for sub_dir in list_imgs:
        if sub_dir in aug_imgs:
            dir_to_move = os.path.join(aug_dir, sub_dir)
            shutil.move(dir_to_move, data_dir)
    """
    dataset_train = FlairDataset()
    dataset_train.load_images(data_dir)
    dataset_train.prepare()
    
    
    dataset_val = FlairDataset()
    dataset_val.load_images(val_dir)
    dataset_val.prepare()
    
    print("Training on " + str(len(dataset_train.image_info)) + " images")
    print("Validating on " + str(len(dataset_val.image_info)) + " images")


        # Validation dataset
    #dataset_val = MRIMathDataset()
    #dataset_val.load_images( '/media/daniel/Backup Data/Flair', 130,180)
    #dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last
    
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        
        model.load_weights(model.find_last()[1], by_name=True)

    
        # Training - Stage 4
    # Fine tune all layers
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE/10,
        epochs=80,
        layers='heads')
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE/10,
        epochs=120,
        layers='all')
    """
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='4+')
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='all')
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=200,
                layers='heads')
    
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=200,
                layers='heads')
    """
    """
    # move the validation data backq
    list_imgs = os.listdir(val_dir)
    for sub_dir in list_imgs:
        dir_to_move = os.path.join(val_dir, sub_dir)
        shutil.move(dir_to_move, data_dir)
    
    """
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
    
