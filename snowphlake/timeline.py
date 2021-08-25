# Author: Vikram Venkatraghavan, Amsterdam UMC

import numpy as np
import pandas as pd
from snowphlake.mixture_model import mixture_model
import snowphlake.mallows_model as mallows_model
import snowphlake.utils as utils
import warnings
from sklearn.mixture import BayesianGaussianMixture as BGM 

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
        self.mixture_model = None 
        self.sequence_model = {'ordering': None, 'event_centers': None,
                        'heterogeneity': None, 'mallows_spread': None}
        self.subtyping_model = {'BayesianModel': [], 'Index': []}
        if self.estimate_uncertainty==True:
            self.bootstrap_mixture_model = [] 
            self.bootstrap_sequence_model = [{'ordering': None, 
                            'event_centers': None, 'heterogeneity': None,
                            'mallows_spread': None} for x in range(N)]
    
    def subtype(self, heterogeneity, diagnosis, p_subtypes0=None):
        
        bgm = BGM(n_components=2, covariance_type = 'diag', random_state = self.random_seed)
        diag_noncn = diagnosis[diagnosis > 1]
        idx_case = diag_noncn == np.nanmax(diagnosis)
        if p_subtypes0 is None:
            bgm.fit(heterogeneity[idx_case,:] )
            p_subtypes = bgm.predict_proba(heterogeneity)
            self.subtyping_model['BayesianModel'].append(bgm)
            self.subtyping_model['Index'].append(1)
        else:
            largest_subtype = np.argmax(np.sum(p_subtypes0,axis=0))
            idx_largestsubtype = np.argmax(p_subtypes0,axis=1) == largest_subtype
            idx_case_select=np.logical_and(idx_largestsubtype, idx_case)
            bgm.fit(heterogeneity[largest_subtype][idx_case_select,:] )
            self.subtyping_model['BayesianModel'].append(bgm)
            self.subtyping_model['Index'].append(largest_subtype)
            p_subtypes = np.zeros((len(diag_noncn),len(self.subtyping_model)+1))

            ##TODO: This is not the right approach. Merge multiple BGM models into a single one.
            p_subtypes[:,:-1] = p_subtypes0
            for k in range(len(self.subtyping_model)):
                if k == largest_subtype:
                    ps = bgm.predict_proba(heterogeneity[largest_subtype][idx_largestsubtype,:])
                    p_subtypes[idx_largestsubtype,k] = ps[:,0]
                    p_subtypes[idx_largestsubtype,-1] = ps[:,1]
                    
        return p_subtypes

    def estimate_instance(self, data, diagnosis, biomarker_labels):

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
        
        mm = mixture_model(data_corrected.shape[1], biomarker_labels,
                    self.n_gaussians, 1, self.random_seed) # Fit it for 1 subtype first
        p_yes = mm.fit(data_corrected,diagnosis)
        pi0, event_centers, ih, indv_mahalanobis, sig0 = mallows_model.weighted_mallows.fitMallows(p_yes[0],1-mm.mixing[:,0])
        # mixture model -> mallows model --> subtyping
        if self.n_maxsubtypes>1:
            p_subtypes = self.subtype(indv_mahalanobis,diagnosis)
            print ('2 subtypes identified. Retuning the timeline parameters.')
            for n_subtypes in range(2,self.n_maxsubtypes):
                p_subtypes0 = np.copy(p_subtypes)
                # mixture model
                mm = mixture_model(data_corrected.shape[1], biomarker_labels,
                        self.n_gaussians, n_subtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis,p_subtypes0)

                # mallows model
                pi0_list = []
                event_centers_list = []
                indv_mahalanobis_list = []
                sig0_list = []
                for k in range(n_subtypes):
                    pi0, event_centers, ih, indv_mahalanobis, sig0 = mallows_model.weighted_mallows.fitMallows(p_yes[k],1-mm.mixing[:,k])
                    pi0_list.append(pi0)
                    event_centers_list.append(event_centers)
                    indv_mahalanobis_list.append(indv_mahalanobis)
                    sig0_list.append(sig0)
                
                # subtyping
                p_subtypes = self.subtype(indv_mahalanobis_list, diagnosis, p_subtypes0)
                print (str(n_subtypes+1)+' subtypes identified. Retuning the timeline parameters.')
                #S = mallows_model()
                #S.fit(p_yes)
            mm = mixture_model(data_corrected.shape[1], biomarker_labels,
                    self.n_gaussians, self.n_maxsubtypes, self.random_seed)
            p_yes = mm.fit(data_corrected,diagnosis,p_subtypes)
            pi0_list = []
            event_centers_list = []
            indv_mahalanobis_list = []
            sig0_list = []
            for k in range(self.n_maxsubtypes):
                pi0, event_centers, ih, indv_mahalanobis, sig0 = mallows_model.weighted_mallows.fitMallows(p_yes[k],1-mm.mixing[:,k])
                pi0_list.append(pi0)
                event_centers_list.append(event_centers)
                indv_mahalanobis_list.append(indv_mahalanobis)
                sig0_list.append(sig0)
        else:
            pi0_list = [pi0]
            event_centers_list = [event_centers]
            indv_mahalanobis_list = [indv_mahalanobis]
            sig0_list = [sig0]
            p_subtypes = None


        return pi0_list, event_centers_list, mm, indv_mahalanobis_list, sig0_list, p_subtypes
    
    def estimate(self, data, diagnosis, biomarker_labels):
        warnings.filterwarnings("ignore")

        pi0,event_centers, mm, indv_mahalanobis, sig0, p_subtypes = self.estimate_instance(data, diagnosis, biomarker_labels)
        self.mixture_model = mm
        self.sequence_model['ordering'] = pi0 
        self.sequence_model['event_centers'] = event_centers
        self.sequence_model['heterogeneity'] = indv_mahalanobis
        self.sequence_model['mallows_spread'] = sig0

        if self.estimate_uncertainty == True:
            for i in range(self.bootstrap_repetitions):
                ## This is not ready yet
                data_resampled, diagnosis_resampled = utils.resample(data,diagnosis,self.random_seed + i)
                pi0_resampled,event_centers_resampled, mm_resampled,
                indv_mahalanobis_resampled, sig0_resampled = estimate_instance(data_resampled, 
                    diagnosis_resampled, biomarker_labels)
                self.bootstrap_mixture_model.append(mm_resampled)
                self.bootstrap_sequence_model[i]['ordering'] = pi0_resampled
                self.bootstrap_sequence_model[i]['event_centers'] = event_centers_resampled 
                self.bootstrap_sequence_model[i]['heterogeneity'] = indv_mahalanobis_resampled
                self.bootstrap_sequence_model[i]['mallows_spread'] = sig0_resampled

        return p_subtypes
    
    def predict_severity(self, data):
        ## This is currently non-functional
        utils.checkifestimated(self)
        data_corrected = self.confounding_factors_model.predict(data)
        p_yes=self.gmm.predict_posterior(data_corrected)
        stages = self.mallows_model.predict(p_yes)

        return stages,p_yes