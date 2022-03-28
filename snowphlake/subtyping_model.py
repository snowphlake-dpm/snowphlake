import os 
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro 
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import STAP
import numpy as np 
from sklearn.model_selection import StratifiedKFold 
import multiprocessing as mp 

class subtyping_model():

    def __init__(self, random_seed = 42, n_maxsubtypes = 1, \
            n_optsubtypes = None, n_nmfruns = 50, \
            subtyping_measure = 'zscore', model_selection = None, n_splits = 5,
            n_cpucores = None):

        self.random_seed = random_seed
        self.n_maxsubtypes = n_maxsubtypes 
        self.n_optsubtypes = n_optsubtypes 
        self.n_nmfruns = n_nmfruns
        self.subtyping_measure = subtyping_measure
        if n_cpucores is not None:
            self.n_cpucores = n_cpucores
        else:
            self.n_cpucores = mp.cpu_count()*2

        self.trained_params = {'Basis': None, 'Theta': None,
                        'normalize': None}

        self.model_selection = model_selection
        if self.model_selection is not None:
            self.rss_data = np.zeros(self.n_maxsubtypes) 
            self.rss_random = np.zeros(self.n_maxsubtypes) 
        if self.subtyping_measure == 'zscore':
            self.params_normalize = None
        self.n_splits = n_splits # used only when model_selection is crossval 

        return
    
    def fit(self, data, diagnosis):
        nmf = importr('NMF')

        def _nmf_call(data_ad_R, n_subtypes, i, n_parallel):
            try:
                model_subtype_opt = nmf.nmf(data_ad_R, n_subtypes, \
                            method='nsNMF', nrun=n_parallel, seed=self.random_seed+i)
                H = np.asarray(ro.r.basis(model_subtype_opt)).copy()
                theta = np.asarray(ro.r.attributes(ro.r.fit(model_subtype_opt))[0])[0].copy()
                self.trained_params['Basis'].append(H)
                self.trained_params['Theta'].append(theta)
                flag_success = 1
            except:
                flag_success = 0
            
            return flag_success            

        def _core_subtyping_module(data_ad, n_subtypes, flag_randomize):
            
            data_ad_R = ro.r.matrix(np.transpose(data_ad),
                nrow=data_ad.shape[1],
                ncol=data_ad.shape[0])
            ro.r.assign("data_ad_R",data_ad_R)
            if flag_randomize == True:
                data_ad_R = nmf.randomize(data_ad_R)
            
            self.trained_params['Basis'] = []
            self.trained_params['Theta'] = []

            remaining_runs = self.n_nmfruns
            cnt = 0
            while remaining_runs > 0:
                n_parallel = np.min([self.n_cpucores,remaining_runs])
                flag_success = _nmf_call(data_ad_R,n_subtypes,cnt,n_parallel)
                cnt = cnt + n_parallel
                if flag_success==1:
                    remaining_runs = remaining_runs - n_parallel
                
            return

        def _subtype_predicting_submodule(data_this, flag_randomize):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(dir_path+'/nmfPredict.R', 'r') as f:
                string = f.read()
            nmfPredict = STAP(string, "predictNMF")

            H_all = self.trained_params['Basis'] 
            theta_all = self.trained_params['Theta']
            
            data_this = np.transpose(data_this)
            data_this_R = ro.r.matrix(data_this,
                nrow=data_this.shape[0],
                ncol=data_this.shape[1])
            ro.r.assign("data_this_R",data_this_R)
            if flag_randomize == True:
                data_this_R = nmf.randomize(data_this_R)
                data_this = np.asarray(data_this_R)

            rss_all = np.zeros(len(H_all))
            for i in range(len(H_all)):
                H = H_all[i]
                theta = theta_all[i]
                prediction_model = nmfPredict.predictNMF(data_this, H, theta)
                weights_R = np.asarray(ro.r.coefficients(prediction_model))
                rss_this = nmf.rss(prediction_model,data_this_R)
                rss_all[i] = np.asarray(rss_this)[0]

            idx_min = np.argmin(rss_all)
            rss = rss_all[idx_min]
            self.trained_params['Basis'] = [H_all[idx_min]]
            self.trained_params['Theta'] = [theta_all[idx_min]]

            return weights_R, rss

        def _core_subtype_predicting_module(data_ad, data_noncn, n_subtypes, flag_randomize, flag_modelselection = False):

            subtypes = np.zeros(diagnosis.shape[0]) + np.nan 
            weight_subtypes = np.zeros((diagnosis.shape[0],n_subtypes)) + np.nan
            
            _, rss_AD = _subtype_predicting_submodule(data_ad, flag_randomize)
            weights_R_noncn, _ = _subtype_predicting_submodule(data_noncn, flag_randomize)
            
            if flag_modelselection is False:
                weight_subtypes[diagnosis!=1,:] = np.transpose(np.asarray(weights_R_noncn))
                subtypes[diagnosis!=1] = np.argmax(weight_subtypes[diagnosis!=1,:],axis=1)

            return subtypes, weight_subtypes, rss_AD

        def _select_opt_n(data,diagnosis):
            print('Evaluating the optimum number of subtypes.')
            for n_subs in range(1,self.n_maxsubtypes+1):
                print ('Checking for N = ', n_subs)
                if self.model_selection == 'full':
                    if self.subtyping_measure == 'zscore':
                        _, _, self.rss_data[n_subs-1] = _zscore_subtyping(data, diagnosis, n_subs, flag_randomize = False)
                        _, _, self.rss_random[n_subs-1] = _zscore_subtyping(data, diagnosis, n_subs, flag_randomize = True)
                    else:
                        _, _, self.rss_data[n_subs-1] = _likelihood_subtyping(data, diagnosis, n_subs, flag_randomize = False)
                        _, _, self.rss_random[n_subs-1] = _likelihood_subtyping(data, diagnosis, n_subs, flag_randomize = True)
                elif self.model_selection == 'crossval':
                        skf = StratifiedKFold(n_splits=5)
                        cnt = -1
                        rss_data_folds = np.zeros(5)
                        rss_random_folds = np.zeros(5)
                        for train_index, val_index in skf.split(data,diagnosis):
                            cnt = cnt+1
                            data_train, data_validate = data[train_index,:], data[val_index,:]
                            diagnosis_train, diagnosis_validate = diagnosis[train_index], diagnosis[val_index]
                            if self.subtyping_measure == 'zscore':
                                _, _, rss_data_folds[cnt] = _zscore_subtyping(data_train, diagnosis_train, n_subs, flag_randomize = False, data_validation = data_validate, diagnosis_validation = diagnosis_validate)
                                _, _, rss_random_folds[cnt] = _zscore_subtyping(data_train, diagnosis_train, n_subs, flag_randomize = True, data_validation = data_validate, diagnosis_validation = diagnosis_validate)
                            else:
                                _, _, self.rss_data[n_subs-1] = _likelihood_subtyping(data_train, diagnosis_train, n_subs, flag_randomize = False, data_validation = data_validate, diagnosis_validation = diagnosis_validate)
                                _, _, self.rss_random[n_subs-1] = _likelihood_subtyping(data_train, diagnosis_train, n_subs, flag_randomize = True, data_validation = data_validate, diagnosis_validation = diagnosis_validate)
                        self.rss_data[n_subs-1] = np.mean(rss_data_folds)
                        self.rss_random[n_subs-1] = np.mean(rss_random_folds)

            flag_more = -np.diff(self.rss_data) > -np.diff(self.rss_random)
            idx_select = np.where(flag_more == True)[0]
            if len(idx_select)==0:
                self.n_optsubtypes = 1
            #elif len(idx_select)>1:
            #    idx_select=idx_select[0]
            else:
                idx_select = idx_select[-1]
                self.n_optsubtypes = range(1,self.n_maxsubtypes+1)[idx_select+1]
            print ('Optimum number of subtypes selected:', self.n_optsubtypes)

            return 
        
        def _likelihood_subtyping(data,diagnosis, n_subtypes,
            flag_randomize = False, data_validation = None, diagnosis_validation = None):
            diagnosis_noncn  = diagnosis[diagnosis!=1]
            idx_ad = diagnosis_noncn == 3

            #data_log = np.log(data+0.001)
            data_ad = data[idx_ad,:]
            #self.trained_params['normalize'] = np.min(data_log)
            #data_ad = data_ad - np.min(data_log)
            _core_subtyping_module(data_ad, n_subtypes, flag_randomize)

            #data_noncn = data_log - self.trained_params['normalize']
            data_noncn = np.copy(data)
            if data_validation is not None:
                idx_ad_val = diagnosis_validation == 3 
                data_ad_val = data_validation[idx_ad_val,:].copy()
                subtypes, weight_subtypes, rss = _core_subtype_predicting_module(data_ad_val, data_noncn, n_subtypes, flag_randomize, flag_modelselection = True)
            else:
                subtypes, weight_subtypes, rss = _core_subtype_predicting_module(data_ad, data_noncn, n_subtypes, flag_randomize)

            return subtypes, weight_subtypes, rss

        def _zscore_subtyping(data, diagnosis, n_subtypes,
            flag_randomize = False, data_validation = None, diagnosis_validation = None):

            idx_ad = diagnosis == 3
            idx_cn = diagnosis == 1
            data_cn = data[idx_cn,:]
            data_ad = data[idx_ad,:]

            mean_cn = np.mean(data_cn,axis=0)
            std_cn = np.std(data_cn,axis=0)
            for i in range(data_cn.shape[1]):
                data_ad[:,i] = -1 * ((data_ad[:,i] - mean_cn[i])/std_cn[i])
            self.trained_params['normalize'] = [mean_cn,std_cn,np.min(data_ad)] 

            data_ad = data_ad - np.min(data_ad)
            _core_subtyping_module(data_ad, n_subtypes, flag_randomize)
            
            data_noncn = data[diagnosis!=1,:] 
            for i in range(data.shape[1]):
                data_noncn[:,i] = -1 * (data_noncn[:,i] - \
                    self.trained_params['normalize'][0][i]) / \
                    self.trained_params['normalize'][1][i]
            data_noncn = data_noncn - self.trained_params['normalize'][2]
            data_noncn[data_noncn<0] = 0

            if data_validation is not None:
                idx_ad_val = diagnosis_validation == 3 
                data_ad_val = data_validation[idx_ad_val,:].copy()
                for i in range(data.shape[1]):
                    data_ad_val[:,i] = -1 * (data_ad_val[:,i] - \
                        self.trained_params['normalize'][0][i]) / \
                        self.trained_params['normalize'][1][i]
                data_ad_val = data_ad_val - self.trained_params['normalize'][2]
                data_ad_val[data_ad_val<0] = 0
                subtypes, weight_subtypes, rss = _core_subtype_predicting_module(data_ad_val, data_noncn, n_subtypes, flag_randomize, flag_modelselection = True)
            else:
                subtypes, weight_subtypes, rss = _core_subtype_predicting_module(data_ad, data_noncn, n_subtypes, flag_randomize)

            return subtypes, weight_subtypes, rss
        
        if self.n_optsubtypes is None:
            _select_opt_n(data,diagnosis)
        
        if self.subtyping_measure == 'zscore':
            subtypes, weight_subtypes, _ = _zscore_subtyping(data,diagnosis, self.n_optsubtypes)
        else:
            subtypes, weight_subtypes, _ = _likelihood_subtyping(data,diagnosis, self.n_optsubtypes)

        return subtypes, weight_subtypes