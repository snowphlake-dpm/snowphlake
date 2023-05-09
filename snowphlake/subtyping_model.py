import os 
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro 
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import STAP
import numpy as np 
from pathos.multiprocessing import ProcessingPool as Pool
import sklearn.metrics as metrics
import nimfa
from sklearn.covariance import MinCovDet

class subtyping_model():

    def __init__(self, random_seed = 42, n_maxsubtypes = 1, \
            n_optsubtypes = None, n_nmfruns = 50, n_nmfruns_perbatch = 10, outlier_percentile = 95, \
            model_selection = None, n_cpucores = None):

        self.outlier_percentile = outlier_percentile
        self.random_seed = random_seed
        self.n_maxsubtypes = n_maxsubtypes 
        self.n_optsubtypes = n_optsubtypes 
        self.n_nmfruns = n_nmfruns
        self.n_nmfruns_perbatch = n_nmfruns_perbatch

        self.trained_params = {'Basis': None, 'Theta': None,
                        'normalize': None, 'Outlier_Detectors': None,
                        'Outlier_threshold': None}

        self.model_selection = model_selection
        if self.model_selection is not None:
            self.rss_data = np.zeros(self.n_maxsubtypes) + np.inf
            self.rss_random = np.zeros(self.n_maxsubtypes) + np.inf
            self.silhouette_score = np.ones(self.n_maxsubtypes)
        else:
            self.rss_data = None 
            self.rss_random = None 
            self.silhouette_score = None

        self.params_normalize = None

        if n_cpucores is not None:
            self.n_cpucores = n_cpucores
        else:
            self.n_cpucores = 1
        return

    def subtype_predicting_submodule(self, data_this, flag_randomize, Basis = None,Theta = None):
        nmf = importr('NMF')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path+'/nmfPredict.R', 'r') as f:
            string = f.read()
        nmfPredict = STAP(string, "predictNMF")

        if Basis == None:
            H_all = self.trained_params['Basis'] 
            theta_all = self.trained_params['Theta']
        else:
            H_all = Basis
            theta_all = Theta
        
        data_this = np.transpose(data_this)
        data_this_R = ro.r.matrix(data_this,
            nrow=data_this.shape[0],
            ncol=data_this.shape[1])
        ro.r.assign("data_this_R",data_this_R)
        if flag_randomize == True:
            data_this_R = nmf.randomize(data_this_R)
            data_this = np.asarray(data_this_R)

        weights_R_all = []
        rss_all = np.zeros(len(H_all))
        for i in range(len(H_all)):
            H = H_all[i]
            theta = theta_all[i]
            prediction_model = nmfPredict.predictNMF(data_this, H, theta)
            weights_R = np.asarray(ro.r.coefficients(prediction_model))
            weights_R_all.append(weights_R)
            rss_this = nmf.rss(prediction_model,data_this_R)
            rss_all[i] = np.asarray(rss_this)[0]
                            
        idx_min = np.argmin(rss_all)
        rss = rss_all[idx_min]
        weights_R = weights_R_all[idx_min]

        return weights_R, rss

    def _nmf_worker(self, inputs):
        data_ad_t, n_subtypes, n_parallel,random_seeds = inputs[0], inputs[1], inputs[2], inputs[3]
        #model_subtype_opt = nmf.nmf(data_ad_R, n_subtypes, \
        #            method='nsNMF', nrun=n_parallel, \
        #            seed=self.random_seed+(n_parallel*i))
        theta = 0.9
        model_subtype_opt = nimfa.Nsnmf(data_ad_t,\
            rank=n_subtypes, max_iter=2000, theta = theta, \
            n_run = n_parallel, random_seeds = random_seeds)
        _ = model_subtype_opt().distance(metric='kl')
        H = model_subtype_opt.basis()
        model_subtype_opt_coef = model_subtype_opt.coef()
        #H = np.asarray(ro.r.basis(model_subtype_opt)).copy()
        #theta = np.asarray(ro.r.attributes(ro.r.fit(model_subtype_opt))[0])[0].copy()
        data_ad = np.transpose(data_ad_t)
        _, rss = self.subtype_predicting_submodule(data_ad, False,[H],[theta])
        
        return H, theta, model_subtype_opt_coef, rss

    def fit(self, data, diagnosis):
        
        def _core_outlierdetection_module(Coefs,labels, n_subtypes):
            outliers = np.zeros(labels.shape,dtype=bool)
            mcd_trained_all = []
            mcd_threshold = []
            for i in range(n_subtypes):
                idx_this = labels == i
                if np.sum(idx_this) > 1:
                    mcd = MinCovDet(random_state=self.random_seed)
                    Coefs = np.asarray(Coefs)
                    Coefs_trans = np.transpose(Coefs)
                    _ = mcd.fit(Coefs_trans[idx_this,:])
                    mb = mcd.mahalanobis(Coefs_trans[idx_this,:])
                    outliers[idx_this] = mb>np.percentile(mb,self.outlier_percentile)
                    thresh = np.percentile(mb,self.outlier_percentile)
                else:
                    mcd = None
                    thresh = None
                mcd_trained_all.append(mcd)
                mcd_threshold.append(thresh)
            return outliers, mcd_trained_all, mcd_threshold

        def _core_subtyping_module(data_ad, n_subtypes, flag_randomize):
            nmf = importr('NMF')

            data_ad_R = ro.r.matrix(np.transpose(data_ad),
                nrow=data_ad.shape[1],
                ncol=data_ad.shape[0])
            ro.r.assign("data_ad_R",data_ad_R)
            if flag_randomize == True:
                data_ad_R = nmf.randomize(data_ad_R)
            self.trained_params['Basis'] = []
            self.trained_params['Theta'] = []
            self.trained_params['Outlier_Detectors'] = []
            self.trained_params['Outlier_threshold'] = []
            
            n_batches = int(self.n_nmfruns / self.n_nmfruns_perbatch)
            prng = np.random.RandomState(self.random_seed)
            random_seeds = prng.choice(self.n_nmfruns*10, size=self.n_nmfruns, replace=False)

            all_inputs = []
            for i in range(n_batches):
                runs = self.n_nmfruns_perbatch
                random_seeds_this = random_seeds[(runs*i):(runs*(i+1))]
                data_ad_t = np.asarray(data_ad_R)
                list_input = [data_ad_t, n_subtypes,runs,random_seeds_this]
                all_inputs.append(list_input)
            
            p = Pool(self.n_cpucores)
            results = p.map(self._nmf_worker, tuple(all_inputs))
                #S = np.asarray(ro.r.silhouette(model_subtype_opt,what='consensus'))
            model_subtype_opt_all = []
            rss_all = []
            for i in range(n_batches):
                H, theta, model_subtype_opt_coef, rss = results[i]
                self.trained_params['Basis'].append(H)
                self.trained_params['Theta'].append(theta)
                model_subtype_opt_all.append(model_subtype_opt_coef)
                rss_all.append(rss)
                #silhouette_score = S[:,-1].mean()
            subtype_metrics = []
            idx_min = np.argmin(np.asarray(rss_all))
            subtype_metrics.append(np.min(np.asarray(rss_all)))
            self.trained_params['Basis'] = [self.trained_params['Basis'][idx_min]]
            self.trained_params['Theta'] = [self.trained_params['Theta'][idx_min]]
            model_subtype_opt_coef = model_subtype_opt_all[idx_min]
            if n_subtypes>1:
                labels = model_subtype_opt_coef.argmax(axis=0)
                labels = np.asarray(labels).flatten()
                outliers, mcd_trained_all, mcd_threshold = _core_outlierdetection_module(model_subtype_opt_coef, labels, n_subtypes)
                silhouette_score = metrics.silhouette_score(data_ad[~outliers,:],labels[~outliers])
            else:
                silhouette_score = 1
                mcd_trained_all = []
                mcd_threshold = []
            self.trained_params['Outlier_Detectors'] = mcd_trained_all
            self.trained_params['Outlier_threshold'] = mcd_threshold

            subtype_metrics.append(silhouette_score)
            #TODO: sort out the silhouette_score issue. The metric is not correct  
            return subtype_metrics

        def _core_subtype_predicting_module(data_noncn, n_subtypes, flag_randomize, flag_modelselection = False):

            subtypes = np.zeros(diagnosis.shape[0]) + np.nan 
            weight_subtypes = np.zeros((diagnosis.shape[0],n_subtypes)) + np.nan
            
            weights_R_noncn, _ = self.subtype_predicting_submodule(data_noncn, flag_randomize)
            
            if flag_modelselection is False:
                weights_transposed = np.transpose(np.asarray(weights_R_noncn))
                weight_subtypes[diagnosis!=1,:] = weights_transposed
                subtypes_withoutliers = np.argmax(weight_subtypes[diagnosis!=1,:],axis=1)
                subtypes_withoutliers = subtypes_withoutliers.astype(float)
                outliers = np.zeros(subtypes_withoutliers.shape,dtype=bool)
                if n_subtypes > 1:
                    for i in range(n_subtypes):
                        idx_this = subtypes_withoutliers == i
                        mcd = self.trained_params['Outlier_Detectors'][i]
                        if mcd is not None:
                            mb = mcd.mahalanobis(weights_transposed[idx_this,:])
                            outliers[idx_this] = mb > self.trained_params['Outlier_threshold'][i]
                    subtypes_withoutliers[outliers] = np.nan
                    subtypes[diagnosis!=1] = subtypes_withoutliers

            return subtypes, weight_subtypes

        def _model_selection_full(n_subs):
            print ('Evaluating for N =', n_subs)
            _, _, metrics = _zscore_subtyping(data, diagnosis, n_subs, flag_randomize = False)
            _, _, metrics_random = _zscore_subtyping(data, diagnosis, n_subs, flag_randomize = True)

            self.rss_data[n_subs-1] = metrics[0]
            self.rss_random[n_subs-1] = metrics_random[0]
            self.silhouette_score[n_subs-1] = metrics[1]
            return

        def _select_opt_n():

            print('Evaluating the optimum number of subtypes.')
            for n_subs in range(1,self.n_maxsubtypes+1):
                if self.model_selection == 'full':
                    _model_selection_full(n_subs)

            max_rand = np.max(-np.diff(self.rss_random))
            flag_more = -np.diff(self.rss_data) > max_rand
            idx_select = np.where(flag_more == True)[0]
            if len(idx_select)==0:
                self.n_optsubtypes = 1
            else:
                idx_select = idx_select[-1]
                self.n_optsubtypes = range(1,self.n_maxsubtypes+1)[idx_select+1]
            print ('Optimum number of subtypes selected:', self.n_optsubtypes)

            return 

        def _zscore_subtyping(data, diagnosis, n_subtypes,
            flag_randomize = False):
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
            subtype_metrics = _core_subtyping_module(data_ad, n_subtypes, flag_randomize)
            
            data_noncn = data[diagnosis!=1,:] 
            for i in range(data.shape[1]):
                data_noncn[:,i] = -1 * (data_noncn[:,i] - \
                    self.trained_params['normalize'][0][i]) / \
                    self.trained_params['normalize'][1][i]
            data_noncn = data_noncn - self.trained_params['normalize'][2]
            data_noncn[data_noncn<0] = 0

            subtypes, weight_subtypes = _core_subtype_predicting_module(data_noncn, n_subtypes, flag_randomize)
            return subtypes, weight_subtypes, subtype_metrics
        
        if self.n_optsubtypes is None:
            _select_opt_n()
        
        print('Estimating subtypes for N =', self.n_optsubtypes)
        subtypes, weight_subtypes, subtype_metrics = _zscore_subtyping(data,diagnosis, self.n_optsubtypes)

        # check if there is an outlier subtype
        #TODO: This causes an error if there are more than 1 outlier subtype
        for i in range(self.n_optsubtypes):
            if np.sum(np.logical_and(subtypes == i,diagnosis==3)) < 0:
                print ('Subtype #'+ str(i) +' is an outlier subtype')
                subtypes[subtypes == i] = np.nan 
                self.n_optsubtypes = self.n_optsubtypes - 1
                self.trained_params['Basis'][0] = np.delete(self.trained_params['Basis'][0],obj=i, axis=1)
                weight_subtypes = np.delete(weight_subtypes,obj=i,axis=1)
        
        if self.rss_data is None:
            self.rss_data = subtype_metrics[0]
            self.silhouette_score = subtype_metrics[1]

        return subtypes, weight_subtypes

    def predict(self, data):
        
        data_norm = data.copy()
        for i in range(data.shape[1]):
            data_norm[:,i] = -1 * (data_norm[:,i] - \
                self.trained_params['normalize'][0][i]) / \
                self.trained_params['normalize'][1][i]
        data_norm = data_norm - self.trained_params['normalize'][2]
        data_norm[data_norm<0] = 0

        weight_subtypes, _ = self.subtype_predicting_submodule(data_norm,False)
        weight_subtypes = np.transpose(weight_subtypes)
        subtypes = np.argmax(weight_subtypes,axis=1)

        return subtypes, weight_subtypes