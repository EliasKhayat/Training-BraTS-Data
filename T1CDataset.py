'''
Created on Jun 18, 2018

@author: daniel
'''
from MRIMathDataset import MRIMathDataset
class T1CDataset(MRIMathDataset):
    def __init__(self):
        super().__init__()
        self.mode = "t1ce"
        self.tumor_type = "active"
    
    def getMask(self, mask):
        mask[mask < 4] = 0
        mask[mask > 0] = 1
        return mask