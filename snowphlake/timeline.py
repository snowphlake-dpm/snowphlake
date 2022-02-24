# Author: Vikram Venkatraghavan, Amsterdam UMC

import numpy as np
from snowphlake.mixture_model import mixture_model
from snowphlake.subtyping_model import subtyping_model
import snowphlake.mallows_model as mallows_model
import snowphlake.utils as utils
import warnings

class timeline():

    def __init__(self,confounding_factors=None, 
                    diagnostic_labels=None, estimate_uncertainty=False, estimate_subtypes = False, 
                    bootstrap_repetitions=100, n_nmfruns = 30, subtyping_measure=None,
                    random_seed=42, n_gaussians = 1, n_maxsubtypes = 1, n_optsubtypes = None):

        self.confounding_factors = confounding_factors
        self.diagnostic_labels = diagnostic_labels 
        self.estimate_uncertainty = estimate_uncertainty
        self.random_seed = random_seed
        self.n_gaussians = n_gaussians
        if self.estimate_uncertainty == True:
            self.bootstrap_repetitions = bootstrap_repetitions
        else:
            self.bootstrap_repetitions = 0
        self.estimate_subtypes = estimate_subtypes
        if self.estimate_subtypes == True:
            if n_optsubtypes is None:
                self.n_optsubtypes = -1 # will be set later
                self.n_maxsubtypes = n_maxsubtypes
            else:
                self.n_optsubtypes = n_optsubtypes
                self.n_maxsubtypes = n_optsubtypes
        else:
            self.n_maxsubtypes = -1 # will be set later
            self.n_optsubtypes = -1 # will be set later
        if subtyping_measure is None:
            self.subtyping_measure='zscore'
        else:
            self.subtyping_measure=subtyping_measure # Can be either zscore or atypicality 
        self.n_nmfruns = n_nmfruns

        self.confounding_factors_model = None 
        self.mixture_model = None 
        self.sequence_model = {'ordering': None, 'event_centers': None,
                        'heterogeneity': None, 'mallows_spread': None}
        self.subtyping_model = None
        self.biomarker_labels = None

        if self.estimate_uncertainty==True:
            self.bootstrap_mixture_model = [] 
            self.bootstrap_subtyping_model = [] 
            self.bootstrap_sequence_model = [{'ordering': None, 
                            'event_centers': None, 'heterogeneity': None,
                            'mallows_spread': None} for x in range(bootstrap_repetitions)]

    def estimate_instance(self, data, diagnosis, subtypes):

        if self.diagnostic_labels is not None:
            diagnosis=utils.set_diagnosis(diagnosis,self.diagnostic_labels)

        if self.confounding_factors is not None:
            #TODO: Add confounding factor correction later
            cf=utils.correct_confounding_factors()
            cf.fit(data,diagnosis,self.confounding_factors)
            self.confounding_factors_model = cf 
            data_corrected = cf.predict(data)
        else:
            data_corrected = data 
        
        sm = []
        w_subtypes = []
        if np.logical_and(self.estimate_subtypes == False, subtypes is None):
            #DEBM
            mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
            p_yes = mm.fit(data_corrected,diagnosis)
            pi0, event_centers, ih, indv_mahalanobis, sig0 = mallows_model.weighted_mallows.fitMallows(p_yes[0],1-mm.mixing[:,0])
        else:
            if np.logical_and(self.estimate_subtypes == True, self.subtyping_measure == 'zscore'):
                # Snowphlake with zscore
                sm = subtyping_model(self.random_seed, self.n_maxsubtypes, self.n_optsubtypes,
                            self.subtyping_measure, self.model_selection)
                subtypes, w_subtypes = sm.fit(data_corrected,diagnosis)
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis, subtypes)
                pi0, event_centers, ih, indv_mahalanobis, sig0 = mallows_model.weighted_mallows.fitMallows(p_yes[0],1-mm.mixing[:,0])
            elif np.logical_and(self.estimate_subtypes == False, subtypes is not None):
                #Co-init DEBM
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis, subtypes)
                pi0 = []
                event_centers = []
                ih = []
                indv_mahalanobis = []
                sig0 = []
                for i in range(self.n_optsubtypes):
                    pi0_i, event_centers_i, ih_i, indv_mahalanobis_i, sig0_i = mallows_model.weighted_mallows.fitMallows(p_yes[i],1-mm.mixing[:,i])
                    pi0.append(pi0_i)
                    event_centers.append(event_centers_i)
                    ih.append(ih_i)
                    indv_mahalanobis.append(indv_mahalanobis_i)
                    sig0.append(sig0_i)

            elif np.logical_and(self.estimate_subtypes == True,self.subtyping_measure == 'atypicality'):
                #Snowphlake with atypicality
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, 1, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis)
                pi0, event_centers, ih, indv_mahalanobis, sig0 = mallows_model.weighted_mallows.fitMallows(p_yes[0],1-mm.mixing[:,0])
                sm = subtyping_model(self.random_seed, self.n_maxsubtypes, self.n_optsubtypes,
                            self.subtyping_measure, self.model_selection)
                subtypes, w_subtypes = sm.fit(indv_mahalanobis,diagnosis)
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis, subtypes)
                pi0, event_centers, ih, indv_mahalanobis, sig0 = mallows_model.weighted_mallows.fitMallows(p_yes[0],1-mm.mixing[:,0])

        return pi0, event_centers, mm, indv_mahalanobis, sig0, sm, w_subtypes
    
    def estimate(self, data, diagnosis, biomarker_labels, subtypes = None):
        warnings.filterwarnings("ignore")
        self.biomarker_labels = biomarker_labels
        if self.estimate_subtypes == True:
            if subtypes is not None:
                print('provided subtypes are being ignored. Set estimate_subtypes = False to use the provided subtypes')
        else:
            if subtypes is None:
                self.n_maxsubtypes = 1 
                self.n_optsubtypes = 1
            else:
                self.n_maxsubtypes = len(np.unique(subtypes[~np.isnan(subtypes)]))
                self.n_optsubtypes = self.n_maxsubtypes

        pi0,event_centers, mm, indv_mahalanobis, sig0, sm, w_subtypes = self.estimate_instance(data, diagnosis, subtypes)
        self.mixture_model = mm
        self.sequence_model['ordering'] = pi0 
        self.sequence_model['event_centers'] = event_centers
        self.sequence_model['heterogeneity'] = indv_mahalanobis
        self.sequence_model['mallows_spread'] = sig0
        self.subtyping_measure = sm 

        if self.estimate_uncertainty == True:
            for i in range(self.bootstrap_repetitions):
                ## This is not ready yet
                data_resampled, diagnosis_resampled = utils.resample(data,diagnosis,self.random_seed + i)
                pi0_resampled,event_centers_resampled, mm_resampled, \
                    indv_mahalanobis_resampled, sig0_resampled, sm_resampled, _ = self.estimate_instance(data_resampled, \
                    diagnosis_resampled, biomarker_labels)
                self.bootstrap_mixture_model.append(mm_resampled)
                self.bootstrap_sequence_model[i]['ordering'] = pi0_resampled
                self.bootstrap_sequence_model[i]['event_centers'] = event_centers_resampled 
                self.bootstrap_sequence_model[i]['heterogeneity'] = indv_mahalanobis_resampled
                self.bootstrap_sequence_model[i]['mallows_spread'] = sig0_resampled
                self.bootstrap_subtyping_model[i] = sm_resampled

        return w_subtypes
    
    def predict_severity(self, data):
        ## This is currently non-functional
        utils.checkifestimated(self)
        data_corrected = self.confounding_factors_model.predict(data)
        p_yes=self.gmm.predict_posterior(data_corrected)
        stages = self.mallows_model.predict(p_yes)

        return stages,p_yes