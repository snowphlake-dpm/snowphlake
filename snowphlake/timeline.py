# Author: Vikram Venkatraghavan, Amsterdam UMC

#TODO: 
# silhouette score in python and R --> compare.
# bootstrap parallelization doesnt work --> Switch from ThreadPool to Pool
# phase out R for randomization and subtype prediction 

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
                    bootstrap_repetitions=100, n_nmfruns = 30, n_nmfruns_perbatch = 10,
                    random_seed=42, n_gaussians = 1,
                    n_maxsubtypes = 1, n_optsubtypes = None, model_selection = None,
                    n_cpucores = None, outlier_percentile = None):

        self.confounding_factors = confounding_factors
        self.diagnostic_labels = diagnostic_labels 
        self.estimate_uncertainty = estimate_uncertainty
        self.random_seed = random_seed
        self.n_gaussians = n_gaussians
        self.outlier_percentile = outlier_percentile 

        if self.estimate_uncertainty == True:
            self.bootstrap_repetitions = bootstrap_repetitions
        else:
            self.bootstrap_repetitions = 0

        if n_optsubtypes==1:
            estimate_subtypes=False

        self.estimate_subtypes = estimate_subtypes

        if self.estimate_subtypes == True:
            self.model_selection = model_selection # full, or None 
            self.n_splits = None
            if self.model_selection is not None:
                n_optsubtypes = None 
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
        self.n_nmfruns = n_nmfruns
        self.n_nmfruns_perbatch = n_nmfruns_perbatch

        if n_cpucores is not None:
            self.n_cpucores = n_cpucores
        else:
            self.n_cpucores = mp.cpu_count()

        self.confounding_factors_model = None 
        self.mixture_model = None 
        self.sequence_model = {'ordering': None, 'event_centers': None, 'event_centers_original_scale': None,
                         'mallows_spread': None, 'variation_predictor': None,
                         'variation_normalization_factor': None}
        self.subtyping_model = None
        self.biomarker_labels = None

        if self.estimate_uncertainty==True:
            self.bootstrap_mixture_model = [[]for x in range(self.bootstrap_repetitions)]
            self.bootstrap_sequence_model = [{'ordering': None, 
                            'event_centers': None,
                            'mallows_spread': None} for x in range(self.bootstrap_repetitions)]
            self.bootstrap_subtyping_model = [[]for x in range(self.bootstrap_repetitions)]

    def scale_event_centers(self,event_centers_i, flag_precomputed):
        
        if flag_precomputed is None:
            org_scale = np.zeros(2)
            org_scale[0] = event_centers_i.min()
            org_scale[1] = event_centers_i.max()
        else:
            org_scale = self.sequence_model['event_centers_original_scale'][flag_precomputed][:]
        
        evn_range = org_scale[1] - org_scale[0]
        scale = 0.9/evn_range
        event_centers_i = (event_centers_i - org_scale[0])*scale + 0.05
        return event_centers_i, org_scale
    
    def scale_atypicality(self, atypicality_sum, subj_stages, flag_precomputed):
        from sklearn.linear_model import RANSACRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.gaussian_process.kernels import ExpSineSquared
        
        atypicality_sum = np.asarray(atypicality_sum)
        if flag_precomputed is None:
            max_atyp = np.max(atypicality_sum)
        else:
            max_atyp = self.sequence_model['variation_normalization_factor'][flag_precomputed]

        atypicality_sum = atypicality_sum / max_atyp
        if flag_precomputed is None:
            reg = RANSACRegressor(estimator=KernelRidge(kernel=ExpSineSquared()),
                          random_state = 0, min_samples = 5)
            reg.fit(subj_stages.reshape(-1,1), atypicality_sum.reshape(-1,1))
        else:
            reg = self.sequence_model['variation_predictor'][flag_precomputed]
        predicted_atyp = reg.predict(subj_stages.reshape(-1,1))
        atypicality_sum = atypicality_sum - predicted_atyp[:,0]

        return atypicality_sum, reg, max_atyp

    def estimate_instance(self, data, diagnosis, subtypes, flag_precomputed = None):
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
            'stage': np.zeros(data.shape[0]) + np.nan ,
            'subtype': np.zeros(data.shape[0]) + np.nan ,
            'subtype_weights': np.zeros(data.shape[0]) + np.nan ,
            'within_subtype_variation': np.zeros(data.shape[0]) + np.nan, 
        }
        
        sm = []
        if np.logical_and(self.estimate_subtypes == False, subtypes is None):
            #DEBM
            mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
            p_yes = mm.fit(data_corrected,diagnosis)
            pi0_i, event_centers_i, atypicality_sum, _, sig0_i = \
                mallows_model.weighted_mallows.fitMallows(p_yes[0],1-mm.mixing[:,0])
            if flag_precomputed is not None:
                flag_precomputed = 0
            event_centers_i, org_scale_i = self.scale_event_centers(event_centers_i, flag_precomputed)
            subj_stages = mallows_model.weighted_mallows.predict_severity(pi0_i,event_centers_i, p_yes[0])
            atypicality_sum, reg, max_atyp = self.scale_atypicality(atypicality_sum, subj_stages, flag_precomputed)

            subjects_derived_info['stage'][diagnosis!=1] = subj_stages
            pi0 = [pi0_i]
            org_scale = [org_scale_i]
            reg_all = [reg]
            max_atyp_all = [max_atyp]
            event_centers = [event_centers_i]
            sig0 = [sig0_i]
            subjects_derived_info['within_subtype_variation'][diagnosis!=1] = np.asarray(atypicality_sum)
        else:
            if self.estimate_subtypes == True:
                # Snowphlake with zscore
                sm = subtyping_model(self.random_seed, self.n_maxsubtypes,
                    self.n_optsubtypes,self.n_nmfruns, self.n_nmfruns_perbatch, self.outlier_percentile,
                    self.model_selection, self.n_cpucores)
                subtypes, w_subtypes = sm.fit(data_corrected,diagnosis)
                self.n_optsubtypes = sm.n_optsubtypes
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis, subtypes)

                pi0 = []
                event_centers = []
                sig0 = []
                org_scale = []
                reg_all = []
                max_atyp_all = []
                subjects_derived_info['subtype'] = subtypes 
                subjects_derived_info['subtype_weights'] = w_subtypes

                unique_subtypes = np.unique(subtypes[~np.isnan(subtypes)])

                for i in range(self.n_optsubtypes):
                    if flag_precomputed is not None:
                        flag_precomputed = i
                    pi0_i, event_centers_i, atypicality_sum_i, atypicality_all_i, sig0_i = \
                        mallows_model.weighted_mallows.fitMallows(p_yes[i],1-mm.mixing[:,i])
                    event_centers_i, org_scale_i = self.scale_event_centers(event_centers_i, flag_precomputed)
                    subj_stages_i = mallows_model.weighted_mallows.predict_severity(pi0_i,event_centers_i, p_yes[i])
                    atypicality_sum_i, reg, max_atyp = self.scale_atypicality(atypicality_sum_i, subj_stages_i, flag_precomputed)

                    subjects_derived_info['stage'][subtypes==unique_subtypes[i]] = subj_stages_i
                    subjects_derived_info['within_subtype_variation'][subtypes==unique_subtypes[i]] = np.asarray(atypicality_sum_i)
                    pi0.append(pi0_i)
                    event_centers.append(event_centers_i)
                    sig0.append(sig0_i)
                    org_scale.append(org_scale_i)
                    reg_all.append(reg)
                    max_atyp_all.append(max_atyp)

            elif np.logical_and(self.estimate_subtypes == False, subtypes is not None):
                #Co-init DEBM
                mm = mixture_model(data_corrected.shape[1],
                        self.n_gaussians, self.n_optsubtypes, self.random_seed)
                p_yes = mm.fit(data_corrected,diagnosis, subtypes)
                pi0 = []
                event_centers = []
                sig0 = []
                org_scale = []
                reg_all = []
                max_atyp_all = []
                subjects_derived_info['subtype'] = subtypes
                subtypes_noncn = subtypes[diagnosis!=1]
                unique_subtypes = np.unique(subtypes_noncn[~np.isnan(subtypes_noncn)]) 
                for i in range(self.n_optsubtypes):
                    if flag_precomputed is not None:
                        flag_precomputed = i
                    pi0_i, event_centers_i, atypicality_sum_i, atypicality_all_i, sig0_i = \
                        mallows_model.weighted_mallows.fitMallows(p_yes[i],1-mm.mixing[:,i])
                    event_centers_i, org_scale_i = self.scale_event_centers(event_centers_i, flag_precomputed)
                    subj_stages_i = mallows_model.weighted_mallows.predict_severity(pi0_i,event_centers_i, p_yes[i])
                    atypicality_sum_i, reg, max_atyp = self.scale_atypicality(atypicality_sum_i, subj_stages_i, flag_precomputed)

                    idx_this = np.logical_and(subtypes==unique_subtypes[i],diagnosis!=1)
                    subjects_derived_info['stage'][idx_this] = subj_stages_i
                    subjects_derived_info['within_subtype_variation'][idx_this] = np.asarray(atypicality_sum_i)
                    pi0.append(pi0_i)
                    event_centers.append(event_centers_i)
                    sig0.append(sig0_i)
                    org_scale.append(org_scale_i)
                    reg_all.append(reg)
                    max_atyp_all.append(max_atyp)

        return pi0, event_centers, mm, sig0, sm, subjects_derived_info, org_scale, reg_all, max_atyp_all
    
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

        pi0,event_centers, mm, sig0, sm, subjects_derived_info, org_scale, reg_all, max_atyp_all = \
            self.estimate_instance(data, diagnosis.copy(), subtypes)
        self.mixture_model = mm
        self.sequence_model['ordering'] = pi0 
        self.sequence_model['event_centers'] = event_centers
        self.sequence_model['mallows_spread'] = sig0
        self.subtyping_model = sm
        self.sequence_model['event_centers_original_scale'] = org_scale
        self.sequence_model['variation_predictor'] = reg_all
        self.sequence_model['variation_normalization_factor'] = max_atyp_all
        print('Timeline estimated.')
        if subtypes is None:
            if self.estimate_subtypes == True:
                subtypes = subjects_derived_info['subtype'].copy()

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

            if estimate_subtypes == True:
                subtypes_resampled = None 

            pi0_resampled,event_centers_resampled, mm_resampled, sig0_resampled, sm_resampled,\
                _, _, _, _ = self.estimate_instance(data_resampled,
                diagnosis_resampled, subtypes_resampled, flag_precomputed=True)

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
            orig_val = self.estimate_subtypes
            self.estimate_subtypes = False
            
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
            self.estimate_subtypes = orig_val
        else:
            subjects_derived_info_resampled = []

        return subjects_derived_info, subjects_derived_info_resampled
    
    def predict(self, data, subtypes = None, unique_subtypes = None, iter_bootstrap = None, diagnosis = None,
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
            'stage': np.zeros(data.shape[0]) + np.nan ,
            'subtype': np.zeros(data.shape[0]) + np.nan ,
            'subtype_weights': np.zeros((data.shape[0],self.n_optsubtypes)) + np.nan ,
            'within_subtype_variation': np.zeros(data.shape[0]) + np.nan, 
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
                subtypes, weights = sub.predict(data_corrected)
                p_yes = mix.predict_posterior(data_corrected, weights)
                subjects_derived_info['subtype_weights'][idx_noncn,:] = weights[idx_noncn,:]
        elif subtypes is not None:
            p_subtypes = np.zeros((subtypes.shape[0],self.n_optsubtypes)) + np.nan 
            if unique_subtypes is None:
                unique_subtypes = np.unique(subtypes[~np.isnan(subtypes)])
            for i in range(len(unique_subtypes)):
                idx_i = subtypes == unique_subtypes[i]
                p_subtypes[idx_i,:] = 0
                p_subtypes[idx_i,i] = 1
                
            p_yes = mix.predict_posterior(data_corrected, p_subtypes)
        else:
            p_yes = mix.predict_posterior(data_corrected, None)

        if subtypes is not None:
            subjects_derived_info['subtype'][idx_noncn] = subtypes[idx_noncn]
            if unique_subtypes is None:
                unique_subtypes = np.unique(subtypes[~np.isnan(subtypes)])
            for i in range(len(unique_subtypes)):
                u = int(unique_subtypes[i])
                pi0_i = seq['ordering'][i]
                event_centers_i = seq['event_centers'][i]
                subj_stages_i, atypicality_sum_i, atypicality_all_i = mallows_model.weighted_mallows.predict(pi0_i,event_centers_i, p_yes[i])
                atypicality_sum_i, _, _ = self.scale_atypicality(atypicality_sum_i, subj_stages_i, i)
                idx_this = subtypes==u
                subjects_derived_info['stage'][idx_this] = subj_stages_i
                subjects_derived_info['within_subtype_variation'][idx_this] = np.asarray(atypicality_sum_i)
            subjects_derived_info['stage'][~idx_noncn] = np.nan 
            subjects_derived_info['within_subtype_variation'][~idx_noncn] = np.nan
        else:
            pi0 = seq['ordering'][0]
            event_centers = seq['event_centers'][0]
            subj_stages, atypicality_sum, atypicality_all = mallows_model.weighted_mallows.predict(pi0,event_centers, p_yes[0])
            atypicality_sum, _, _ = self.scale_atypicality(atypicality_sum, subj_stages, 0)
            subjects_derived_info['stage'][idx_noncn] = subj_stages[idx_noncn]
            subjects_derived_info['within_subtype_variation'][idx_noncn] = np.asarray(atypicality_sum)[idx_noncn]

        return subjects_derived_info