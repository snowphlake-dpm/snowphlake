# Author: Vikram Venkatraghavan, Amsterdam UMC

import numpy as np
import pandas as pd
import sklearn
import scipy as sp 
from snowphlake.subtyping import subtype
import snowphlake.mallows_model as mallows_model
import snowphlake.utils as utils
import time 
import warnings
class timeline():

    def __init__(self,confounding_factors=None, 
                    diagnostic_labels=None, estimate_uncertainty=False, bootstrap_repetitions=100,
                    random_seed=42, n_gaussians = 1, n_maxsubtypes = 1):

        self.confounding_factors = confounding_factors
        self.diagnostic_labels = diagnostic_labels 
        self.estimate_uncertainty = estimate_uncertainty
        self.random_seed = random_seed
        self.n_gaussians = n_gaussians
        self.n_maxsubtypes = n_maxsubtypes
        self.bootstrap_repetitions = bootstrap_repetitions
        
        self.confounding_factors_model = None 
        self.subtyping_model = None 
        self.sequence_model = {'ordering': None, 'event_centers': None}
        if self.estimate_uncertainty==True:
            self.bootstrap_subtyping_model = [] 
            self.bootstrap_sequence_model = [{'ordering': None, 
                            'event_centers': None} for x in range(N)] 
    
    def estimate(self, data, diagnosis, biomarker_labels):
        warnings.filterwarnings("ignore")
        def estimate_instance(data, diagnosis, biomarker_labels):

            if self.diagnostic_labels is not None:
                #TODO: map diagnostic_labels to integers.
                diagnosis=utils.set_diagnosis(diagnosis,self.diagnostic_labels)

            if self.confounding_factors is not None:
                #TODO: Add confounding factor correction later
                cf=utils.correct_confounding_factors()
                cf.fit(data,diagnosis,self.confounding_factors)
                self.confounding_factors_model = cf 
                data_corrected = cf.predict(data)
            else:
                data_corrected = data 
            
            st = subtype(data_corrected.shape[1], biomarker_labels,
                        self.n_gaussians, self.n_maxsubtypes, self.random_seed)
            p_yes = st.fit(data_corrected,diagnosis)

            from pyebm.central_ordering import generalized_mallows as gm
            pi0_list = []
            event_centers_list = []
            for k in range(self.n_maxsubtypes):
                pi0,event_centers,ih = gm.weighted_mallows.fitMallows(p_yes[k],1-st.mixing[:,k])
                pi0_list.append(pi0)
                event_centers_list.append(event_centers)
            #S = mallows_model()
            #S.fit(p_yes)

            return pi0_list, event_centers_list, st

        pi0,event_centers, st = estimate_instance(data, diagnosis, biomarker_labels)
        self.subtyping_model = st
        self.sequence_model['ordering'] = pi0 
        self.sequence_model['event_centers'] = event_centers

        if self.estimate_uncertainty == True:
            for i in range(self.bootstrap_repetitions):
                ## This is not ready yet
                data_resampled, diagnosis_resampled = utils.resample(data,diagnosis,self.random_seed + i)
                pi0_resampled,event_centers_resampled, st_resampled = estimate_instance(data_resampled, diagnosis_resampled, 
                                biomarker_labels)
                self.bootstrap_subtyping_model.append(st_resampled)
                self.bootstrap_sequence_model[i]['ordering'] = pi0_resampled
                self.bootstrap_sequence_model[i]['event_centers'] = event_centers_resampled 

        return
    
    def predict_severity(self, data):

        utils.checkifestimated(self)
        data_corrected = self.confounding_factors_model.predict(data)
        p_yes=self.gmm.predict_posterior(data_corrected)
        stages = self.mallows_model.predict(p_yes)

        return stages,p_yes