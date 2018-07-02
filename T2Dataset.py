'''
Created on Jun 18, 2018

@author: daniel
'''
from MRIMathDataset import MRIMathDataset
class T2Dataset(MRIMathDataset):
        
    def __init__(self):
        super().__init__()

        self.mode = "t2"
        self.tumor_type = "core"
    
    def getMask(self, mask):
        
        mask[mask == 2] = 0
        mask[mask > 0] = 1
        return mask