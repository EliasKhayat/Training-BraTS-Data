'''
Created on May 29, 2018

@author: daniel
'''
import os
import nibabel as nib


import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import random
import scipy
ia.seed(1)


def addGaussianNoise(image):
    row,col= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

seq = iaa.Sequential([
    #iaa.Fliplr(0.5), # horizontal flips
    #iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.25)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    #iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.ElasticTransformation(alpha=(0.0, 2.0), sigma=0.25),
    #iaa.Multiply((0.5, 1.5)),

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #rotate=(-3, 3),
        #shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order


data_dir = "Data/BRATS_2018/HGG"
for subdir in os.listdir(data_dir):
    os.mkdir("Data/BRATS_2018/HGG_Rot/rot_" + subdir)
    theta = random.randint(-30,30)
    for path in os.listdir(data_dir + "/" + subdir):
        img_rot = np.zeros((240,240,155))
        nib_img = nib.load(data_dir + "/" + subdir + "/" + path)
        np_nib_img = nib_img.get_data()
        for j in range(0,155):
            img = np_nib_img[:,:,j]
            #img = addGaussianNoise(img)
            img_rot[:,:,j] = scipy.ndimage.interpolation.rotate(img, theta, reshape=False)
            #img_aug[:,:,j] = seq.augment_image(img)
        nib.save(nib.Nifti1Image(img_rot, nib_img.affine), "Data/BRATS_2018/HGG_Rot/rot_" + subdir + "/" + "rot_" + path)

            
