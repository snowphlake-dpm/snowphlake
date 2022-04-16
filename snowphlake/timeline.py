# Author: Vikram Venkatraghavan, Amsterdam UMC

#TODO: 
# predict atypicality measure for debm, co-init debm.
# predict function untested for likelihood based subtyping.
# get all nmf runs and store in list.
# get cophenetic score and silhouette score in testing.
# bootstrap parallelization doesnt work --> Switch from ThreadPool to Pool
# bootstrap predict crash for EDADS dataset

import numpy as np
from snowphlake.mixture_model import mixture_model
from snowphlake.subtyping_model import subtyping_model
import snowphlake.mallows_model as mallows_model
import snowphlake.utils as utils
import warnings
import multiprocessing as mp 
from multiprocessing.pool import ThreadPool as Pool

class timeline():

    def __init__(self,confounding_factors=None, diagnostic_labels=None,
                    estimate_uncertainty=False, estimate_subtypes = False, 
                    bootstrap_repetitions=100, n_nmfruns = 30, 
                    subtyping_measure=None, random_seed=42, n_gaussians = 1,
                    n_maxsubtypes = 1, n_optsubtypes = None, model_selection = None,
                    n_splits = 10, n_cpucores = None):

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

        if n_cpucores is not None:
            self.n_cpucores = n_cpucores
        else:
            self.n_cpucores = mp.cpu_count()

        self.confounding_factors_model = None 
        self.mixture_model = None 
        self.sequence_model = {'ordering': None, 'event_centers': None,
                         'mallows_spread': None}
        self.subtyping_model = None
        self.biomarker_labels = None

        if self.estimate_uncertainty==True:
            self.bootstrap_mixture_model = [[]for x in range(self.bootstrap_repetitions)]
            self.bootstrap_sequence_model = [{'ordering': None, 
                            'event_centers': None,
                            'mallows_spread': None} for x in range(self.bootstrap_repetitions)]
            self.bootstrap_subtyping_model = [[]for x in range(self.bootstrap_repetitions)]

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
                    self.model_selection, self.n_splits, self.n_cpucores)
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
                    subjects_derived_info['atypicality'][subtypes==i] = np.asarray(atypicality_sum_i)
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
                    subjects_derived_info['atypicality'][idx_this] = np.asarray(atypicality_sum_i)
                    subjects_derived_info['atypicality_all'][idx_this,:] = atypicality_all_i
                    pi0.append(pi0_i)
                    event_centers.append(event_centers_i)
                    sig0.append(sig0_i)

            elif np.logical_and(self.estimate_subtypes == True,\
                    self.subtyping_measure == 'likelihood'):
                #Snowphlake with likelihood
                mm_init = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, 1, self.random_seed)
                p_yes = mm_init.fit(data_corrected,diagnosis,get_likelihood = True)
                sm = subtyping_model(self.random_seed, self.n_maxsubtypes,
                    self.n_optsubtypes,self.n_nmfruns, self.subtyping_measure,
                    self.model_selection, self.n_splits, self.n_cpucores)
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
                    subjects_derived_info['atypicality'][subtypes==i] = np.asarray(atypicality_sum_i)
                    subjects_derived_info['atypicality_all'][subtypes==i,:] = atypicality_all_i
                    
                    pi0.append(pi0_i)
                    event_centers.append(event_centers_i)
                    sig0.append(sig0_i)
                mm = [mm_init, mm]

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
            self.estimate_instance(data, diagnosis.copy(), subtypes)
        self.mixture_model = mm
        self.sequence_model['ordering'] = pi0 
        self.sequence_model['event_centers'] = event_centers
        self.sequence_model['mallows_spread'] = sig0
        self.subtyping_model = sm 
        print('Timeline estimated.')
        if subtypes is None:
            if self.estimate_subtypes == True:
                subtypes = subjects_derived_info['subtypes'].copy()

        def _reorder_subtype_params(i):
            H = self.subtyping_model.trained_params['Basis'][0]
            Hi = self.bootstrap_subtyping_model[i].trained_params['Basis'][0].copy()

            basis_dist = np.zeros((H.shape[1],H.shape[1]))
            for j in range(H.shape[1]):
                for k in range(Hi.shape[1]):
                    basis_dist[j,k] = np.sqrt(np.sum((H[:,j] - Hi[:,k])**2))

            idx_map = np.argmin(basis_dist,axis=1)

            for j in range(Hi.shape[1]):
                self.bootstrap_subtyping_model[i].trained_params['Basis'][0][:,j] = Hi[:,idx_map[j]]

            for j in range(len(self.biomarker_labels)):
                mu = self.bootstrap_mixture_model[i].cases[j]['mu'].copy()
                self.bootstrap_mixture_model[i].cases[j]['mu'] = mu[:,idx_map]
                std = self.bootstrap_mixture_model[i].cases[j]['std'].copy()
                self.bootstrap_mixture_model[i].cases[j]['std'] = std[:,idx_map]
                w = self.bootstrap_mixture_model[i].cases[j]['weights'].copy()
                self.bootstrap_mixture_model[i].cases[j]['weights'] = w[:,idx_map]

            for j in range(Hi.shape[1]):
                ordering = self.bootstrap_sequence_model[i]['ordering'].copy()
                event_centers = self.bootstrap_sequence_model[i]['event_centers'].copy()
                mallows_spread = self.bootstrap_sequence_model[i]['mallows_spread'].copy()

                for k in range(H.shape[1]):
                    self.bootstrap_sequence_model[i]['ordering'][k] = ordering[idx_map[k]].copy()
                    self.bootstrap_sequence_model[i]['event_centers'][k] = event_centers[idx_map[k]].copy()
                    self.bootstrap_sequence_model[i]['mallows_spread'][k] = mallows_spread[idx_map[k]].copy()
            
            return

        def _estimate_uncertainty(i,estimate_subtypes):
            data_resampled, diagnosis_resampled, subtypes_resampled = \
                utils.bootstrap_resample(data,diagnosis.copy(),subtypes,self.random_seed + i)

            pi0_resampled,event_centers_resampled, mm_resampled, sig0_resampled, sm_resampled,\
                _ = self.estimate_instance(data_resampled,
                diagnosis_resampled, subtypes_resampled)

            self.bootstrap_mixture_model[i] = mm_resampled
            self.bootstrap_subtyping_model[i] = sm_resampled
            self.bootstrap_sequence_model[i]['ordering'] = pi0_resampled
            self.bootstrap_sequence_model[i]['event_centers'] = event_centers_resampled 
            self.bootstrap_sequence_model[i]['mallows_spread'] = sig0_resampled

            if estimate_subtypes == True:
                _reorder_subtype_params(i)

            subjects_derived_info_resampled_i = timeline.predict(self, data,\
                subtypes = subtypes, iter_bootstrap = i, estimate_subtypes = estimate_subtypes, diagnosis = diagnosis)            
            return subjects_derived_info_resampled_i

        def _strore_result(output):
            subjects_derived_info_resampled = [[] for x in range(self.bootstrap_repetitions)]
            for x in range(len(output)):
                subjects_derived_info_resampled[x] = output[x].copy()
            return subjects_derived_info_resampled

        if self.estimate_uncertainty == True:
            print('Estimating uncertainty.')
            #orig_val = self.estimate_subtypes
            #self.estimate_subtypes = False
            
            #pool = Pool(self.n_cpucores * 2)
            #inputs = []
            outputs = []
            for i in range(self.bootstrap_repetitions):
                #inputs.append([i,orig_val])
                print([i],end='', flush=True)
                output_this = _estimate_uncertainty(i, self.estimate_subtypes)
                outputs.append(output_this)
            #outputs = pool.starmap(_estimate_uncertainty, inputs)
            subjects_derived_info_resampled = _strore_result(list(outputs))
            #pool.close()
            #pool.join()
            #self.estimate_subtypes = orig_val
        else:
            subjects_derived_info_resampled = []

        return subjects_derived_info, subjects_derived_info_resampled
    
    def predict(self, data, subtypes = None, iter_bootstrap = None, diagnosis = None,
        estimate_subtypes = False):
        ## This is currently non-functional
        if iter_bootstrap is None:
            cf = self.confounding_factors_model
            mix = self.mixture_model
            sub = self.subtyping_model
            seq = self.sequence_model
        else:
            cf = self.confounding_factors_model
            mix = self.bootstrap_mixture_model[iter_bootstrap]
            sub = self.bootstrap_subtyping_model[iter_bootstrap]
            seq = self.bootstrap_sequence_model[iter_bootstrap]
        subjects_derived_info = {
            'staging': np.zeros(data.shape[0]) + np.nan ,
            'subtypes': np.zeros(data.shape[0]) + np.nan ,
            'subtypes_weights': np.zeros((data.shape[0],self.n_optsubtypes)) + np.nan ,
            'atypicality': np.zeros(data.shape[0]) + np.nan, 
            'atypicality_all': np.zeros((data.shape[0],data.shape[1]-1)) + np.nan
        }
        if self.confounding_factors is not None:
            #TODO: confounding factor predict function 
            data_corrected = cf.predict(data)
        else:
            data_corrected = data 
        
        if diagnosis is not None:
            if self.diagnostic_labels is not None:
                diagnosis=utils.set_diagnosis(diagnosis,self.diagnostic_labels)
            idx_noncn = diagnosis != 1
        else:
            idx_noncn = np.ones(data.shape[0],dtype=bool)
        if type(mix) == list:
            p_yes = mix[0].predict_posterior(data_corrected, None, get_likelihood = True)
        if estimate_subtypes == True:
            if self.subtyping_measure=='zscore':
                subtypes, weights = sub.predict(data_corrected)
                p_yes = mix.predict_posterior(data_corrected, weights)
            else:
                subtypes, weights = sub.predict(p_yes)
                p_yes = mix[1].predict_posterior(data_corrected, weights)
            subjects_derived_info['subtypes_weights'][idx_noncn,:] = weights[idx_noncn,:]
        elif subtypes is not None:
            p_subtypes = np.zeros((subtypes.shape[0],self.n_optsubtypes)) + np.nan 
            unique_subtypes = np.unique(subtypes[~np.isnan(subtypes)])
            for i in range(self.n_optsubtypes):
                idx_i = subtypes == unique_subtypes[i]
                p_subtypes[idx_i,:] = 0
                p_subtypes[idx_i,i] = 1
                
            p_yes = mix.predict_posterior(data_corrected, p_subtypes)
        else:
            p_yes = mix.predict_posterior(data_corrected, None)

        if subtypes is not None:
            subjects_derived_info['subtypes'][idx_noncn] = subtypes[idx_noncn]
            unique_subtypes = np.unique(subtypes[~np.isnan(subtypes)])
            for i in range(len(unique_subtypes)):
                pi0_i = seq['ordering'][unique_subtypes[i]]
                event_centers_i = seq['event_centers'][unique_subtypes[i]]
                subj_stages_i, atypicality_sum_i, atypicality_all_i = mallows_model.weighted_mallows.predict(pi0_i,event_centers_i, p_yes[unique_subtypes[i]])
                idx_this = subtypes==unique_subtypes[i]
                subjects_derived_info['staging'][idx_this] = subj_stages_i
                #TODO: Atypicality estimated in training and testing are not the same. Rectify this.
                #TODO: Remove atypicality_all
                subjects_derived_info['atypicality'][idx_this] = np.asarray(atypicality_sum_i)
                subjects_derived_info['atypicality_all'][idx_this,:] = atypicality_all_i[:,1:-1]
            subjects_derived_info['staging'][~idx_noncn] = np.nan 
            subjects_derived_info['atypicality'][~idx_noncn] = np.nan
            subjects_derived_info['atypicality_all'][~idx_noncn,:] = np.nan 
        else:
            pi0 = seq['ordering'][0]
            event_centers = seq['event_centers'][0]
            subj_stages, atypicality_sum, atypicality_all = mallows_model.weighted_mallows.predict(pi0,event_centers, p_yes[0])
            subjects_derived_info['staging'][idx_noncn] = subj_stages[idx_noncn]
            subjects_derived_info['atypicality'][idx_noncn] = np.asarray(atypicality_sum)[idx_noncn]
            subjects_derived_info['atypicality_all'][idx_noncn,:] = atypicality_all[:,1:-1][idx_noncn,:]

        return subjects_derived_info