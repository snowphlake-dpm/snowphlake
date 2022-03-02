# Author: Vikram Venkatraghavan, Amsterdam UMC

import pandas as pd 
import numpy as np 
from sklearn.utils import resample

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

def bootstrap_resample(data,diagnosis,subtypes,random_seed):

    #TODO: Check stratification if diagnosis has nan values 

    if subtypes is not None:
        data_resampled, diagnosis_resampled, subtypes_resampled = \
            resample(data,diagnosis,subtypes, random_state = random_seed, \
                stratify = diagnosis)
    else:
        data_resampled, diagnosis_resampled = \
            resample(data,diagnosis, random_state = random_seed, \
                stratify = diagnosis)
        subtypes_resampled = None 
        
    return data_resampled, diagnosis_resampled, subtypes_resampled