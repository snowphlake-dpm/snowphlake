# Author: Vikram Venkatraghavan, Amsterdam UMC

import numpy as np
import pandas as pd
import sklearn
import scipy as sp 
from scipy import stats
from scipy import optimize 
import multiprocessing

# Need scipy 1.7.1 or higher

class subtype():

    def __init__(self, N, biomarker_labels, n_gaussians=1, n_maxsubtypes=1,
                    random_seed=42):
        # N is the number of events in the model.

        self.n_gaussians = n_gaussians
        self.n_maxsubtypes = n_maxsubtypes
        self.random_seed = random_seed
        self.biomarker_labels = biomarker_labels

        self.controls = [{"mu":np.zeros(self.n_gaussians) ,"std":np.zeros(self.n_gaussians), 
                        "weights": np.zeros(self.n_gaussians)+1/n_gaussians} for x in range(N)]
        self.cases = [{"mu":np.zeros((self.n_gaussians,self.n_maxsubtypes)),
                    "std":np.zeros((self.n_gaussians,self.n_maxsubtypes)), 
                        "weights": np.zeros((self.n_gaussians,self.n_maxsubtypes)) + 1/n_gaussians} for x in range(N)] 
        self.mixing = np.zeros((N,self.n_maxsubtypes)) + 0.5
    
    def mixture_model(self, data_corrected, diagnosis, p_subtypes):

        def _calculate_total_likelihood(data, controls, cases, mixing, subtype_ind):
            
            #subtype_ind = int(subtype_ind)
            total_likeli = np.zeros(data.shape[0])
            likeli_norm = np.zeros(data.shape[0])
            likeli_abnorm = np.zeros(data.shape[0])
            for j in range(self.n_gaussians):
                dist_norm=stats.norm(loc = controls['mu'][j], 
                                    scale = controls['std'][j])
                ## This weighted addition for multiple Gaussians is questionable
                likeli_norm[:] = likeli_norm[:] + (dist_norm.pdf(data) * controls['weights'][j])
                dist_abnorm = stats.norm(loc = cases['mu'][j,subtype_ind], scale = cases['std'][j,subtype_ind])
                likeli_abnorm[:] = likeli_abnorm[:] + (dist_abnorm.pdf(data) * cases['weights'][j,subtype_ind])

            
            total_likeli = (likeli_abnorm[:]*mixing) + (likeli_norm[:] * (1-mixing))
            
            return total_likeli

        def _objective_mixing(mixing,data,cases,controls,p_subtypes_k,subtype_ind):
            total_likeli = _calculate_total_likelihood(data, 
                                        controls, cases, mixing, subtype_ind)
                                        
            objfun = -np.sum(np.log(total_likeli) + np.log(p_subtypes_k))
            #objfun = -np.sum(np.log(total_likeli))
            
            return objfun
    
        def _objective_cases_distribution(cases, data, controls, mixing, p_subtypes_k,subtype_ind):

            cases_unflatten = {"mu":np.zeros((self.n_gaussians,self.n_maxsubtypes)),
                    "std":np.zeros((self.n_gaussians,self.n_maxsubtypes)), 
                        "weights": np.zeros((self.n_gaussians,self.n_maxsubtypes)) + 1/self.n_gaussians}
            cases_unflatten['mu'][0,subtype_ind]=cases[0]
            cases_unflatten['std'][0,subtype_ind]=cases[1]
            total_likeli = _calculate_total_likelihood(data, 
                                        controls, cases_unflatten, mixing, subtype_ind)
            objfun = -np.sum(np.log(total_likeli) + np.log(p_subtypes_k))
            #objfun = -np.sum(np.log(total_likeli))

            return objfun 
        
        N = data_corrected.shape[1]
        idx_cn = diagnosis == 1 
        idx_cases = diagnosis == np.nanmax(diagnosis)
        flag_opt_stop=0
        cnt = 0
        data_noncn = data_corrected[~idx_cn,:]
        self.mixing[:,:]=0.5
        while flag_opt_stop==0:
            cnt = cnt+1
            mixing0 = np.copy(self.mixing)
            if cnt==1:
                mixing_init = np.copy(mixing0)
            
            for k in range(self.n_maxsubtypes):
                idx_select = p_subtypes[:,k]>=0.5
                for i in range(N):
                    bnd_mixing = np.asarray([0.05,0.95])
                    bnd_mixing = np.repeat(bnd_mixing[np.newaxis,:],1,axis=0)
                    res=optimize.minimize(_objective_mixing,mixing0[i,k],
                                    args=(data_noncn[idx_select,i],self.cases[i],self.controls[i],p_subtypes[idx_select,k],k),
                                    method='SLSQP', bounds=bnd_mixing)
                    self.mixing[i,k] = res.x
            
            if np.mean(np.abs(self.mixing-mixing0)) < 0.01:
                print('mixing:',np.mean(np.abs(self.mixing-mixing0)))
                flag_opt_stop=1
                break 
            else:
                print('mixing:',np.mean(np.abs(self.mixing-mixing0)))

            ## Check for diverging mixing parameter in small datasets  to introduce sanity check

            for k in range(self.n_maxsubtypes):
                idx_select = p_subtypes[:,k]>=0.5

                for i in range(N):

                    cases0 = np.asarray([self.cases[i]['mu'][:,k].flatten(), self.cases[i]['std'][:,k].flatten()])
                    ## Excluding weights for now in this optimization
                    ## Check the order of flattening when introducing subtypes 
                    ## Also when introducing multiple Gaussians

                    bnd_cases = np.zeros((2,2))
                    if np.mean(self.cases[i]['mu'][:])<np.mean(self.controls[i]['mu'][:]):
                        bnd_cases[0,:] = np.asarray([np.nanmin(data_corrected[idx_cases,i]),
                                                    self.controls[i]['mu'][0]])
                        
                    else:
                        bnd_cases[0,:] = np.asarray([self.controls[i]['mu'][0],
                                                    np.nanmax(data_corrected[idx_cases,i])])

                    bnd_cases[1,:] = np.asarray([0,np.std(data_corrected[idx_cases,i])])

                    res=optimize.minimize(_objective_cases_distribution,cases0,
                                args=(data_noncn[idx_select,i],self.controls[i],self.mixing[i,k], p_subtypes[idx_select,k],k),
                                method='SLSQP',bounds=bnd_cases)
                    
                    ## Check this too while introducing subtypes and multiple Gaussians
                    self.cases[i]['mu'][0,k]=res.x[0]
                    self.cases[i]['std'][0,k]=res.x[1]
                    ##self.cases[i]['weights'][0,0]=res.x[2]

                    # --> do a sanity check for rejecting the optimization
                    # if GMM fails, revert to initialized values
        return
    
    def cluster_model(self, p_yes_list, diagnosis, p_subtypes0):
        ## Change this to clustering based on subject-specific orderings.
        from sklearn.decomposition import LatentDirichletAllocation as LDA
        idx_cases = diagnosis == np.nanmax(diagnosis)
        lda = LDA(self.n_maxsubtypes, random_state = self.random_seed)
        p_yes_k = np.zeros((p_yes_list[0].shape[0],p_yes_list[0].shape[1],len(p_yes_list)))
        for k in range(self.n_maxsubtypes):
            for i in range(p_yes_k.shape[1]):
                p_yes_k[:,i,k]=np.multiply(p_yes_list[k][:,i], p_subtypes0[:,k])
        p_yes = np.sum(p_yes_k,axis=2)
        lda.fit(p_yes[idx_cases,:],diagnosis[idx_cases])
        p_subtypes = lda.transform(p_yes)
        return p_subtypes
    
    def fit(self, data_corrected, diagnosis):
        
        ## First implementation for n_gaussians=1 and n_maxsubtypes=1
        # Truncated Gaussian for initialization 
        N = data_corrected.shape[1]

        idx_cn = diagnosis == 1 
        idx_cases = diagnosis == np.nanmax(diagnosis)
        for i in range(N):
            # Initial Fit
            self.controls[i]['mu'][0],self.controls[i]['std'][0]=stats.norm.fit(data_corrected[idx_cn,i])
            self.cases[i]['mu'][0,:], self.cases[i]['std'][0,:]=stats.norm.fit(data_corrected[idx_cases,i])
            
            # Reject overlapping regions 
            likeli_norm = stats.norm.pdf(data_corrected[idx_cases,i], 
                loc = self.controls[i]['mu'][0], scale = self.controls[i]['std'][0])
            likeli_abnorm = stats.norm.pdf(data_corrected[idx_cases,i], 
                loc = self.cases[i]['mu'][0,0], scale = self.cases[i]['std'][0,0])
            idx_reject = likeli_norm > likeli_abnorm

            # Truncated Fit --> make this for multiple gaussians
            self.cases[i]['mu'][0,:], self.cases[i]['std'][0,:]= \
                stats.norm.fit(data_corrected[idx_cases,i][~idx_reject])
        
        # Optimization
        flag_stop=0
        cnt = 0
        p_subtypes = np.zeros((data_corrected[diagnosis!=1,:].shape[0],self.n_maxsubtypes)) + 1/self.n_maxsubtypes
        while flag_stop==0:
            cnt = cnt + 1
            print(cnt)
            self.mixture_model(data_corrected, diagnosis, p_subtypes)
            p_yes_list = self.predict_posterior(data_corrected[diagnosis!=1,:])
            if self.n_maxsubtypes > 1:
                p_subtypes0 = np.copy(p_subtypes)
                if cnt == 1:
                    p_subtypes_init  = p_subtypes0
                p_subtypes = self.cluster_model(p_yes_list,diagnosis[diagnosis != 1], p_subtypes0)
                np.save('p_subtypes.npy',[p_subtypes,p_subtypes0])
                if np.mean(np.abs(p_subtypes - p_subtypes0))< (0.01*self.n_maxsubtypes):
                    flag_stop=1
                if cnt == 2:
                    flag_stop=1
                else:
                    print('subtyping:', np.mean(np.abs(p_subtypes - p_subtypes0)))
            else:
                flag_stop=1

        return p_yes_list
    
    def predict_posterior(self,data_corrected):
        
        N = data_corrected.shape[1]
        p_yes = np.zeros((data_corrected.shape[0],N))
        p_yes_list = []
        for k in range(self.n_maxsubtypes):
            for i in range(N):

                wlikeli_norm = (1-self.mixing[i,k])*stats.norm.pdf(data_corrected[:,i], 
                    loc = self.controls[i]['mu'][0], scale = self.controls[i]['std'][0])
                wlikeli_abnorm = (self.mixing[i,k])*stats.norm.pdf(data_corrected[:,i], 
                    loc = self.cases[i]['mu'][0,k], scale = self.cases[i]['std'][0,k])

                p_yes[:,i] = np.divide(wlikeli_abnorm , (wlikeli_abnorm + wlikeli_norm))
            p_yes_list.append(p_yes)
            
        return p_yes_list
