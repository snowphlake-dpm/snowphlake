# Author: Vikram Venkatraghavan, Amsterdam UMC

import pandas as pd 
import numpy as np 

def set_diagnosis(diagnosis,labels):

    idx_cn = diagnosis == labels[0]
    idx_ad = diagnosis == labels[-1]

    diagnosis[idx_cn]=1
    diagnosis[idx_ad]=3
    diagnosis[~np.logical_or(idx_cn,idx_ad)]=2

    return diagnosis

class correct_confounding_factors():

    def fit(self,data,diagnosis,confounding_factors):

        return
    
    def predict(self,data):
        
        return

def resample(data,diagnosis,random_seed):

    return