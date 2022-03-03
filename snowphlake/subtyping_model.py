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
            self.rss_data = np.zeros(self.n_maxsubtypes-1) 
            self.rss_random = np.zeros(self.n_maxsubtypes-1) 
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
        
        def _core_subtype_predicting_module(data_noncn, diagnosis, n_subtypes, flag_randomize):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(dir_path+'/nmfPredict.R', 'r') as f:
                string = f.read()
            nmfPredict = STAP(string, "predictNMF")
            H = self.trained_params['Basis'] 
            theta = self.trained_params['Theta']
            subtypes = np.zeros(data.shape[0]) + np.nan 
            weight_subtypes = np.zeros((data.shape[0],n_subtypes)) + np.nan
            
            data_noncn = np.transpose(data_noncn)
            data_noncnR = ro.r.matrix(data_noncn,
                nrow=data_noncn.shape[0],
                ncol=data_noncn.shape[1])
            ro.r.assign("data_noncnR",data_noncnR)
            if flag_randomize == True:
                data_noncnR = nmf.randomize(data_noncnR)
                data_noncn = np.asarray(data_noncnR)

            prediction_model = nmfPredict.predictNMF(data_noncn, H, theta)
            weights_R = np.asarray(ro.r.coefficients(prediction_model))

            rss = nmf.rss(prediction_model,data_noncnR)
            rss = np.asarray(rss)[0]

            weight_subtypes[diagnosis!=1,:] = np.transpose(np.asarray(weights_R))
            subtypes[diagnosis!=1] = np.argmax(weight_subtypes[diagnosis!=1,:],axis=1)

            return subtypes, weight_subtypes, rss

        def _select_opt_n(data,diagnosis):
            print('Evaluating the optimum number of subtypes.')
            for n_subs in range(2,self.n_maxsubtypes+1):
                print ('Checking for N = ', n_subs)
                if self.model_selection == 'full':
                    _, _, self.rss_data[n_subs-2] = _zscore_subtyping(data, diagnosis, n_subs, flag_randomize = False)
                    _, _, self.rss_random[n_subs-2] = _zscore_subtyping(data, diagnosis, n_subs, flag_randomize = True)

            flag_more = -np.diff(self.rss_data) > -np.diff(self.rss_random)
            idx_select = np.where(flag_more == False)[0]
            if len(idx_select)==0:
                idx_select=-1
            elif len(idx_select)>1:
                idx_select=idx_select[-1]
            else:
                idx_select = idx_select[0]

            self.n_optsubtypes = range(2,self.n_maxsubtypes+1)[idx_select]
            print ('Optimum number of subtypes selected:', self.n_optsubtypes)

            return 
        
        def _atypicality_subtyping(data,diagnosis):
            return 0,0

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

            subtypes, weight_subtypes, rss = _core_subtype_predicting_module(data_noncn, diagnosis, n_subtypes, flag_randomize)

            return subtypes, weight_subtypes, rss
        
        if self.n_optsubtypes is None:
            _select_opt_n(data,diagnosis)
        
        if self.subtyping_measure == 'zscore':
            subtypes, weight_subtypes, _ = _zscore_subtyping(data,diagnosis, self.n_optsubtypes)
        else:
            subtypes, weight_subtypes = _atypicality_subtyping(data,diagnosis)

        return subtypes, weight_subtypes