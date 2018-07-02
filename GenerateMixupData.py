'''
Created on May 31, 2018

@author: daniel
'''

import os
import nibabel as nib
import numpy as np
import random

a = 0.2
data_dir = "Data/BRATS_2018/HGG"
list_imgs = os.listdir(data_dir)
random.shuffle(list_imgs)
        
for i in range(0, len(list_imgs)-1):
    os.mkdir("Data/BRATS_2018/HGG_Mixup/mixup_" + list_imgs[i])
    path_1 = os.listdir(data_dir + "/" + list_imgs[i])
    path_2 = os.listdir(data_dir + "/" + list_imgs[i+1])
    flair_1 = next((s for s in path_1 if "flair" in s), None)
    flair_2 = next((s for s in path_2 if "flair" in s), None)
    nib_img_1 = nib.load(data_dir + "/" + path_1 + "/" + flair_1)
    nib_img_2 = nib.load(data_dir + "/" + path_2 + "/" + flair_2)
    x_i = nib_img_1.get_data()
    x_j = nib_img_1.get_data()
    for j in range(0,155):
        lam = np.random.beta(a)
        img = lam*x_i[:,:,j] + (1-lam)*x[:,:,j]
        nib.save(nib.Nifti1Image(img, nib_img.affine), "Data/BRATS_2018/HGG_Aug/aug_" + subdir + "/" + "aug_" + path)
