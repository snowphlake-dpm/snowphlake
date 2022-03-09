import os 
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro 
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import STAP
import numpy as np 
from sklearn.model_selection import train_test_split 

class subtyping_model():

    def __init__(self, random_seed = 42, n_maxsubtypes = 1, \
            n_optsubtypes = None, n_nmfruns = 30, \
            subtyping_measure = 'zscore', model_selection = None):

        self.random_seed = random_seed
        self.n_maxsubtypes = n_maxsubtypes 
        self.n_optsubtypes = n_optsubtypes 
        self.n_nmfruns = n_nmfruns
        self.subtyping_measure = subtyping_measure

        self.trained_params = {'Basis': None, 'Theta': None,
                        'normalize': None}

        self.model_selection = model_selection
        if self.model_selection is not None:
            self.rss_data = np.zeros(self.n_maxsubtypes) 
            self.rss_random = np.zeros(self.n_maxsubtypes) 
        if self.subtyping_measure == 'zscore':
            self.params_normalize = None 

        return
    
    def fit(self, data, diagnosis):
        nmf = importr('NMF')

        def _core_subtyping_module(data_ad, n_subtypes, flag_randomize):
                    
            data_ad_R = ro.r.matrix(np.transpose(data_ad),
                nrow=data_ad.shape[1],
                ncol=data_ad.shape[0])
            ro.r.assign("data_ad_R",data_ad_R)
            if flag_randomize == True:
                data_ad_R = nmf.randomize(data_ad_R)
                
            model_subtype_opt = nmf.nmf(data_ad_R, n_subtypes, \
                method='nsNMF', nrun=self.n_nmfruns, seed=self.random_seed)

            H = np.asarray(ro.r.basis(model_subtype_opt))

            self.trained_params['Basis'] = H 
            theta = np.asarray(ro.r.attributes(ro.r.fit(model_subtype_opt))[0])[0]
            self.trained_params['Theta'] = theta

            return

        def _subtype_predicting_submodule(data_this, flag_randomize):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(dir_path+'/nmfPredict.R', 'r') as f:
                string = f.read()
            nmfPredict = STAP(string, "predictNMF")

            H = self.trained_params['Basis'] 
            theta = self.trained_params['Theta']
            
            data_this = np.transpose(data_this)
            data_this_R = ro.r.matrix(data_this,
                nrow=data_this.shape[0],
                ncol=data_this.shape[1])
            ro.r.assign("data_this_R",data_this_R)
            if flag_randomize == True:
                data_this_R = nmf.randomize(data_this_R)
                data_this = np.asarray(data_this_R)

            prediction_model = nmfPredict.predictNMF(data_this, H, theta)
            weights_R = np.asarray(ro.r.coefficients(prediction_model))

            rss = nmf.rss(prediction_model,data_this_R)
            rss = np.asarray(rss)[0]

            return weights_R, rss

        def _core_subtype_predicting_module(data_ad, data_noncn, n_subtypes, flag_randomize):

            subtypes = np.zeros(diagnosis.shape[0]) + np.nan 
            weight_subtypes = np.zeros((diagnosis.shape[0],n_subtypes)) + np.nan
            
            weights_R_noncn, _ = _subtype_predicting_submodule(data_noncn, flag_randomize)
            _, rss_AD = _subtype_predicting_submodule(data_ad, flag_randomize)

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
        
        def _likelihood_subtyping(data,diagnosis, n_subtypes, flag_randomize = False):
            diagnosis_noncn  = diagnosis[diagnosis!=1]
            idx_ad = diagnosis_noncn == 3
            data_ad = data[idx_ad,:]
            _core_subtyping_module(data_ad, n_subtypes, flag_randomize)

            data_noncn = data
            subtypes, weight_subtypes, rss = _core_subtype_predicting_module(data_ad, data_noncn, n_subtypes, flag_randomize)

            return subtypes, weight_subtypes, rss

        def _zscore_subtyping(data, diagnosis, n_subtypes, flag_randomize = False):

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

            subtypes, weight_subtypes, rss = _core_subtype_predicting_module(data_ad, data_noncn, n_subtypes, flag_randomize)

            return subtypes, weight_subtypes, rss
        
        if self.n_optsubtypes is None:
            _select_opt_n(data,diagnosis)
        
        if self.subtyping_measure == 'zscore':
            subtypes, weight_subtypes, _ = _zscore_subtyping(data,diagnosis, self.n_optsubtypes)
        else:
            subtypes, weight_subtypes, _ = _likelihood_subtyping(data,diagnosis, self.n_optsubtypes)

        return subtypes, weight_subtypes
