'''
Created on Jun 18, 2018

@author: daniel
'''
from mrcnn.config import Config

class MRIMathConfig(Config):
    # Give the configuration a recognizable name
    NAME = "mrimath"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 2 shapes
        # Number of training steps per epoch
    #STEPS_PER_EPOCH = 200

    # Skip detections with < 90% confidence
    #DETECTION_MIN_CONFIDENCE = 0.0

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    
    IMAGE_MAX_DIM = 256
    #LEARNING_RATE = 0.0001
    #LEARNING_RATE = 0.00001

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 128


    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1,
        "mrcnn_mask_loss": 1.1
    }
    #BACKBONE = "resnet50"

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20