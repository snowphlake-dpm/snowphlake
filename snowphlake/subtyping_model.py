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

    def __init__(self, random_seed = 42, n_maxsubtypes = 1, n_optsubtypes = None, n_nmfruns = 30, subtyping_measure = 'zscore'):

        self.random_seed = random_seed
        self.n_maxsubtypes = n_maxsubtypes 
        self.n_optsubtypes = n_optsubtypes 
        self.n_nmfruns = n_nmfruns
        self.subtyping_measure = subtyping_measure
        self.Basis = None 
        self.Theta = None 
        if self.subtyping_measure == 'zscore':
            self.params_normalize = None 

        return
    
    def fit(self, data, diagnosis):

        if self.n_optsubtypes is None:
            _select_opt_n(data,diagnosis)
        
        if self.subtyping_measure == 'zscore':
            _zscore_subtyping(data,diagnosis)
        else:
            _atypicality_subtyping(data,diagnosis)
        
    
        def _select_opt_n():
            return 
        
        def _atypicality_subtyping():
            return

        def _zscore_subtyping(self, data, diagnosis):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(dir_path+'/nmfPredict.R', 'r') as f:
                string = f.read()
            nmfPredict = STAP(string, "predictNMF")
            nmf = importr('NMF')

            idx_ad = diagnosis == 3
            idx_cn = diagnosis == 1
            data_cn = data[idx_cn,:]
            data_ad = data[idx_ad,:]

            mean_cn = np.mean(data_cn,axis=0)
            std_cn = np.std(data_cn,axis=0)
            for i in range(data_cn.shape[1]):
                data_ad[:,i] = -1 * ((data_ad[:,i] - mean_cn[i])/std_cn[i])
            self.subtyping_model['params_normalize'] = [mean_cn,std_cn,np.min(data_ad)] 

            data_ad = data_ad - np.min(data_ad)
            
            rss_data = np.zeros(self.n_maxsubtypes-1)
            rss_rand = np.zeros(self.n_maxsubtypes-1)
            for n_subs in range(2,self.n_maxsubtypes+1):
                print(n_subs)
                # split train and validate for model selection 
                data_ad_train,data_ad_validate=train_test_split(data_ad,test_size=0.2, random_state = self.random_seed)
                data_ad_train_R = ro.r.matrix(np.transpose(data_ad_train),
                    nrow=data_ad_train.shape[1],
                    ncol=data_ad_train.shape[0])
                ro.r.assign("data_ad_train_R",data_ad_train_R)
                model_subtype = nmf.nmf(data_ad_train_R,n_subs,method='nsNMF', nrun=self.n_nmfruns, seed=self.random_seed)
                
                data_ad_validate_R = ro.r.matrix(np.transpose(data_ad_validate),
                    nrow=data_ad_validate.shape[1],
                    ncol=data_ad_validate.shape[0])
                ro.r.assign("data_ad_validate_R",data_ad_validate_R)

                model_subtype_prediction = nmfPredict.predictNMF(data_ad_validate_R,model_subtype)
                rss_data[n_subs-2] = np.asarray(nmf.rss(model_subtype_prediction,data_ad_validate_R))[0]

                rand_data_ad_train_R= nmf.randomize(data_ad_train_R)
                model_subtype_rand = nmf.nmf(rand_data_ad_train_R, n_subs,method='nsNMF', nrun=self.n_nmfruns, seed=self.random_seed)
                rand_data_ad_validate_R= nmf.randomize(data_ad_validate_R)
                model_subtype_rand_prediction = nmfPredict.predictNMF(rand_data_ad_validate_R,model_subtype_rand)
                rss_rand[n_subs-2] = np.asarray(nmf.rss(model_subtype_rand_prediction,rand_data_ad_validate_R))[0]
            
            flag_more = -np.diff(rss_data) > -np.diff(rss_rand)
            idx_select = np.where(flag_more == False)[0]
            if len(idx_select)==0:
                idx_select=-1
            elif len(idx_select)>1:
                idx_select=idx_select[-1]
            opt_subtype = range(2,self.n_maxsubtypes+1)[idx_select]
            data_ad_R = ro.r.matrix(np.transpose(data_ad),
                    nrow=data_ad.shape[1],
                    ncol=data_ad.shape[0])
            ro.r.assign("data_ad_R",data_ad_R)
            model_subtype_opt = nmf.nmf(data_ad_R, opt_subtype, method='nsNMF', nrun=self.n_nmfruns, seed=self.random_seed)

            W = np.asarray(ro.r.basis(model_subtype_opt))
            weight_subtypes_ad = np.asarray(ro.r.coefficients(model_subtype_opt))
            self.subtyping_model['Basis'] = W 
            # set theta
            weight_subtypes = weight_subtypes_ad 

            return weight_subtypes

        return 