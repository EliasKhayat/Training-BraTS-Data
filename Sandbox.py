'''
Created on May 25, 2018

@author: daniel
'''
'''
Created on May 7, 2018
@author: daniel
'''
import os
import nibabel as nib
import matplotlib .pyplot as plt

flair_aug_path = "Data/BRATS_2018/HGG_Rot/rot_Brats18_2013_2_1/rot_Brats18_2013_2_1_flair.nii.gz"

#seg_path = "Data/BRATS_2018/HGG_Rot/rot_Brats18_2013_2_1/rot_Brats18_2013_3_1_seg.nii.gz"
flair_path = "Data/BRATS_2018/HGG/Brats18_2013_2_1/Brats18_2013_2_1_flair.nii.gz"
seg_aug_path = "Data/BRATS_2018/HGG_Rot/rot_Brats18_2013_2_1/rot_Brats18_2013_2_1_seg.nii.gz"

N = 100
seg_img = nib.load(flair_aug_path)
flair_img = nib.load(seg_aug_path)
seg_img = seg_img.get_data()
flair_img = flair_img.get_data()
print(seg_img.shape)
seg_img = seg_img[:,:,N]
flair_img = flair_img[:,:,N]
plt.figure(1)
plt.subplot(211)
plt.imshow(flair_img)
#super_threshold_indices = img < 2
#img[super_threshold_indices] = 0
#img.reshape(img.shape[0], img.shape[1])
plt.subplot(212)

plt.imshow(seg_img)
plt.show() 