# Author: Vikram Venkatraghavan, Amsterdam UMC

import numpy as np
from snowphlake.mixture_model import mixture_model
from snowphlake.subtyping_model import subtyping_model
import snowphlake.mallows_model as mallows_model
import snowphlake.utils as utils
import warnings

class timeline():

    def __init__(self,confounding_factors=None, diagnostic_labels=None,
                    estimate_uncertainty=False, estimate_subtypes = False, 
                    bootstrap_repetitions=100, n_nmfruns = 50, 
                    subtyping_measure=None, random_seed=42, n_gaussians = 1,
                    n_maxsubtypes = 1, n_optsubtypes = None, model_selection = None,
                    n_splits = 5):

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
            self.model_selection = model_selection # crossval, full, or None 
            self.n_splits = None
            if self.model_selection is not None:
                n_optsubtypes = None 
                self.n_splits = n_splits
            else:
                if n_optsubtypes is None: 
                    print('Error: If model_selection is None, specify n_optsubtypes')
            if n_optsubtypes is None:
                self.n_optsubtypes = None # will be set later
                self.n_maxsubtypes = n_maxsubtypes
            else:
                self.n_optsubtypes = n_optsubtypes
                self.n_maxsubtypes = n_optsubtypes
        else:
            self.n_maxsubtypes = None # will be set later
            self.n_optsubtypes = None # will be set later
        if subtyping_measure is None:
            self.subtyping_measure='zscore'
        else:
            self.subtyping_measure=subtyping_measure # Can be either zscore or likelihood 
        self.n_nmfruns = n_nmfruns

        self.confounding_factors_model = None 
        self.mixture_model = None 
        self.sequence_model = {'ordering': None, 'event_centers': None,
                         'mallows_spread': None}
        self.subtyping_model = None
        self.biomarker_labels = None

        if self.estimate_uncertainty==True:
            self.bootstrap_mixture_model = [] 
            self.bootstrap_subtyping_model = [] 
            self.bootstrap_sequence_model = [{'ordering': None, 
                            'event_centers': None,
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

        subjects_derived_info = {
            'staging': np.zeros(data.shape[0]) + np.nan ,
            'subtypes': np.zeros(data.shape[0]) + np.nan ,
            'subtypes_weights': np.zeros(data.shape[0]) + np.nan ,
            'atypicality': np.zeros(data.shape[0]) + np.nan, 
            'atypicality_all': np.zeros((data.shape[0],data.shape[1]-1)) + np.nan
        }
        
        sm = []
        if np.logical_and(self.estimate_subtypes == False, subtypes is None):
            #DEBM
            mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
            p_yes = mm.fit(data_corrected,diagnosis)
            pi0_i, event_centers_i, atypicality_sum, atypicality_all, sig0_i = \
                mallows_model.weighted_mallows.fitMallows(p_yes[0],1-mm.mixing[:,0])
            subj_stages = mallows_model.weighted_mallows.predict_severity(pi0_i,event_centers_i, p_yes[0])

            subjects_derived_info['staging'][diagnosis!=1] = subj_stages
            pi0 = [pi0_i]
            event_centers = [event_centers_i]
            sig0 = [sig0_i]
            subjects_derived_info['atypicality'][diagnosis!=1] = np.asarray(atypicality_sum)
            subjects_derived_info['atypicality_all'][diagnosis!=1,:] = atypicality_all
        else:
            if np.logical_and(self.estimate_subtypes == True, \
                    self.subtyping_measure == 'zscore'):
                # Snowphlake with zscore
                sm = subtyping_model(self.random_seed, self.n_maxsubtypes,
                    self.n_optsubtypes,self.n_nmfruns, self.subtyping_measure,
                    self.model_selection, self.n_splits)
                subtypes, w_subtypes = sm.fit(data_corrected,diagnosis)
                self.n_optsubtypes = sm.n_optsubtypes
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis, subtypes)

                pi0 = []
                event_centers = []
                sig0 = []
                subjects_derived_info['subtypes'] = subtypes 
                subjects_derived_info['subtypes_weights'] = w_subtypes

                for i in range(self.n_optsubtypes):
                    pi0_i, event_centers_i, atypicality_sum_i, atypicality_all_i, sig0_i = \
                        mallows_model.weighted_mallows.fitMallows(p_yes[i],1-mm.mixing[:,i])
                    subj_stages_i = mallows_model.weighted_mallows.predict_severity(pi0_i,event_centers_i, p_yes[i])

                    subjects_derived_info['staging'][subtypes==i] = subj_stages_i
                    subjects_derived_info['atypicality'][subtypes==i] = atypicality_sum_i
                    subjects_derived_info['atypicality_all'][subtypes==i,:] = atypicality_all_i
                    pi0.append(pi0_i)
                    event_centers.append(event_centers_i)
                    sig0.append(sig0_i)

            elif np.logical_and(self.estimate_subtypes == False, subtypes is not None):
                #Co-init DEBM
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis, subtypes)
                pi0 = []
                event_centers = []
                sig0 = []
                subjects_derived_info['subtypes'] = subtypes
                subtypes_noncn = subtypes[diagnosis!=1]
                unique_subtypes = np.unique(subtypes_noncn[~np.isnan(subtypes_noncn)]) 
                for i in range(self.n_optsubtypes):
                    pi0_i, event_centers_i, atypicality_sum_i, atypicality_all_i, sig0_i = \
                        mallows_model.weighted_mallows.fitMallows(p_yes[i],1-mm.mixing[:,i])
                    subj_stages_i = mallows_model.weighted_mallows.predict_severity(pi0_i,event_centers_i, p_yes[i])

                    idx_this = np.logical_and(subtypes==unique_subtypes[i],diagnosis!=1)
                    subjects_derived_info['staging'][idx_this] = subj_stages_i
                    subjects_derived_info['atypicality'][idx_this] = atypicality_sum_i
                    subjects_derived_info['atypicality_all'][idx_this,:] = atypicality_all_i
                    pi0.append(pi0_i)
                    event_centers.append(event_centers_i)
                    sig0.append(sig0_i)

            elif np.logical_and(self.estimate_subtypes == True,\
                    self.subtyping_measure == 'likelihood'):
                #Snowphlake with atypicality
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, 1, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis,get_likelihood = True)
                sm = subtyping_model(self.random_seed, self.n_maxsubtypes,
                    self.n_optsubtypes,self.n_nmfruns, self.subtyping_measure,
                    self.model_selection, self.n_splits)
                subtypes, w_subtypes = sm.fit(p_yes[0],diagnosis)
                self.n_optsubtypes = sm.n_optsubtypes
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis, subtypes)
                pi0 = []
                event_centers = []
                sig0 = []
                subjects_derived_info['subtypes'] = subtypes 
                subjects_derived_info['subtypes_weights'] = w_subtypes
                for i in range(self.n_optsubtypes):
                    pi0_i, event_centers_i, atypicality_sum_i, atypicality_all_i, sig0_i = \
                        mallows_model.weighted_mallows.fitMallows(p_yes[i],1-mm.mixing[:,i])
                    subj_stages_i = mallows_model.weighted_mallows.predict_severity(pi0_i,event_centers_i, p_yes[i])

                    subjects_derived_info['staging'][subtypes==i] = subj_stages_i
                    subjects_derived_info['atypicality'][subtypes==i] = atypicality_sum_i
                    subjects_derived_info['atypicality_all'][subtypes==i,:] = atypicality_all_i
                    
                    pi0.append(pi0_i)
                    event_centers.append(event_centers_i)
                    sig0.append(sig0_i)

        return pi0, event_centers, mm, sig0, sm, subjects_derived_info
    
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

        pi0,event_centers, mm, sig0, sm, subjects_derived_info = \
            self.estimate_instance(data, diagnosis, subtypes)
        self.mixture_model = mm
        self.sequence_model['ordering'] = pi0 
        self.sequence_model['event_centers'] = event_centers
        self.sequence_model['mallows_spread'] = sig0
        self.subtyping_model = sm 
        subjects_derived_info_resampled = []
        if self.estimate_uncertainty == True:
            for i in range(self.bootstrap_repetitions):
                data_resampled, diagnosis_resampled, subtypes_resampled = \
                    utils.bootstrap_resample(data,diagnosis,subtypes,self.random_seed + i)
                #TODO: For data-driven subtyping, make sure the subtypes are rearranged so that 
                # each subtype id belongs to the closest from the main model.
                pi0_resampled,event_centers_resampled, mm_resampled,\
                    sig0_resampled, sm_resampled, \
                    subjects_derived_info_resampled_i = self.estimate_instance(data_resampled, \
                            diagnosis_resampled, subtypes_resampled)
                
                subjects_derived_info_resampled.append(subjects_derived_info_resampled_i)
                self.bootstrap_mixture_model.append(mm_resampled)
                self.bootstrap_sequence_model[i]['ordering'] = pi0_resampled
                self.bootstrap_sequence_model[i]['event_centers'] = event_centers_resampled 
                self.bootstrap_sequence_model[i]['mallows_spread'] = sig0_resampled
                self.bootstrap_subtyping_model.append(sm_resampled)

        return subjects_derived_info, subjects_derived_info_resampled
    
    def predict_severity(self, data):
        ## This is currently non-functional
        utils.checkifestimated(self)
        data_corrected = self.confounding_factors_model.predict(data)
        p_yes=self.gmm.predict_posterior(data_corrected)
        stages = self.mallows_model.predict(p_yes)

        return stages,p_yes