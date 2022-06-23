# Author: Vikram Venkatraghavan, Amsterdam UMC

import numpy as np
import pandas as pd
import sklearn
import scipy as sp 
from scipy import stats
from scipy import optimize 
import multiprocessing

# Need scipy 1.7.1 or higher

class mixture_model():

    def __init__(self, N, n_gaussians=1, n_optsubtypes=1,
                    random_seed=42):
        # N is the number of events in the model.

        self.n_gaussians = n_gaussians
        self.n_optsubtypes = n_optsubtypes
        self.random_seed = random_seed

        self.controls = [{"mu":np.zeros(self.n_gaussians) ,"std":np.zeros(self.n_gaussians), 
                        "weights": np.zeros(self.n_gaussians)+1/n_gaussians} for x in range(N)]
        self.cases = [{"mu":np.zeros((self.n_gaussians,self.n_optsubtypes)),
                    "std":np.zeros((self.n_gaussians,self.n_optsubtypes)), 
                        "weights": np.zeros((self.n_gaussians,self.n_optsubtypes)) + 1/n_gaussians} for x in range(N)] 
        self.mixing = np.zeros((N,self.n_optsubtypes)) + 0.5
    
    def maximum_likelihood_estimate(self, data_corrected, diagnosis, p_subtypes):

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

            cases_unflatten = {"mu":np.zeros((self.n_gaussians,self.n_optsubtypes)),
                    "std":np.zeros((self.n_gaussians,self.n_optsubtypes)), 
                        "weights": np.zeros((self.n_gaussians,self.n_optsubtypes)) + 1/self.n_gaussians}
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
        data_noncn = data_corrected[~idx_cn,:]
        #self.mixing[:,:]=0.5
        if p_subtypes is None:
            p_subtypes = np.ones((data_noncn.shape[0],self.n_optsubtypes))
       
        for k in range(self.n_optsubtypes):
            idx_select = np.logical_and(np.max(p_subtypes,axis=1)>0,np.argmax(p_subtypes,axis=1)==k)
            for i in range(N):
                cnt = 0
                flag_opt_stop = 0
                while flag_opt_stop==0:
                    cnt = cnt+1
                    mixing0 = np.copy(self.mixing)
                    if cnt==1:
                        mixing_init = np.copy(mixing0)
                    bnd_mixing = np.asarray([0.01,0.99])
                    bnd_mixing = np.repeat(bnd_mixing[np.newaxis,:],1,axis=0)
                    dnc_i = data_noncn[idx_select,i]
                    idx_notnan = ~np.isnan(dnc_i)
                    dnc_i = dnc_i[idx_notnan]
                    pk = p_subtypes[idx_select,k]
                    pk = pk[idx_notnan]
                    res=optimize.minimize(_objective_mixing,mixing0[i,k],
                                    args=(dnc_i,self.cases[i],self.controls[i],pk,k),
                                    method='SLSQP', bounds=bnd_mixing)
                    self.mixing[i,k] = res.x
            
                    if np.abs(self.mixing[i,k]-mixing0[i,k]) < 0.01:
                        #print('mixing convergence check:',np.mean(np.abs(self.mixing[i,k]-mixing0[i,k])))
                        flag_opt_stop=1
                        break 
                    #else:
                        #print('mixing convergence check:',np.mean(np.abs(self.mixing[i,k]-mixing0[i,k])))

                    ## Check for diverging mixing parameter in small datasets  to introduce sanity check

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
                dnc_i = data_noncn[idx_select,i]
                idx_notnan = ~np.isnan(dnc_i)
                dnc_i = dnc_i[idx_notnan]
                pk = p_subtypes[idx_select,k]
                pk = pk[idx_notnan]
                res=optimize.minimize(_objective_cases_distribution,cases0,
                            args=(dnc_i,self.controls[i],self.mixing[i,k], pk,k),
                            method='SLSQP',bounds=bnd_cases)
                    
                ## Check this too while introducing subtypes and multiple Gaussians
                self.cases[i]['mu'][0,k]=res.x[0]
                self.cases[i]['std'][0,k]=res.x[1]
                ##self.cases[i]['weights'][0,0]=res.x[2]

                # --> do a sanity check for rejecting the optimization
                # if GMM fails, revert to initialized values
        return
    
    
    def fit(self, data_corrected, diagnosis, subtypes=None, get_likelihood = False):
        
        ## First implementation for n_gaussians=1 and n_maxsubtypes=1
        # Truncated Gaussian for initialization 

        N = data_corrected.shape[1]

        idx_cn = diagnosis == 1 
        idx_cases = diagnosis == np.nanmax(diagnosis)

        if subtypes is None:
            p_subtypes = None 
        else:
            subtypes_noncn = subtypes[~idx_cn]
            p_subtypes = np.zeros((subtypes_noncn.shape[0],self.n_optsubtypes))
            unique_subtypes = np.unique(subtypes_noncn[~np.isnan(subtypes_noncn)])
            for i in range(self.n_optsubtypes):
                idx_i = subtypes_noncn == unique_subtypes[i]
                p_subtypes[idx_i,i] = 1

        for i in range(N):
            # Initial Fit
            dci_cn = data_corrected[idx_cn,i]
            dci_cn = dci_cn[~np.isnan(dci_cn)]
            dci_cases = data_corrected[idx_cases,i]
            dci_cases = dci_cases[~np.isnan(dci_cases)]

            self.controls[i]['mu'][0],self.controls[i]['std'][0]=stats.norm.fit(dci_cn)
            self.cases[i]['mu'][0,:], self.cases[i]['std'][0,:]=stats.norm.fit(dci_cases)
            
            # Reject overlapping regions 
            likeli_norm = stats.norm.pdf(dci_cases, 
                loc = self.controls[i]['mu'][0], scale = self.controls[i]['std'][0])
            likeli_abnorm = stats.norm.pdf(dci_cases, 
                loc = self.cases[i]['mu'][0,0], scale = self.cases[i]['std'][0,0])
            idx_reject = likeli_norm > likeli_abnorm

            # Truncated Fit --> make this for multiple gaussians
            self.cases[i]['mu'][0,:], self.cases[i]['std'][0,:]= \
                stats.norm.fit(dci_cases[~idx_reject])

        # Optimization
        self.maximum_likelihood_estimate(data_corrected, diagnosis, p_subtypes)
        p_yes_list = self.predict_posterior(data_corrected[diagnosis!=1,:],p_subtypes,get_likelihood)
        
        return p_yes_list
    
    def predict_posterior(self,data_corrected,p_subtypes, get_likelihood = False):
        
        N = data_corrected.shape[1]
        if p_subtypes is None:
            p_subtypes = np.ones((data_corrected.shape[0],1))
        p_yes_list = []

        if get_likelihood == False:
            mixing = np.copy(self.mixing)
        else:
            mixing = 0.5 + np.zeros(self.mixing.shape)

        for k in range(self.n_optsubtypes):
            
            idx_select = np.logical_and(np.argmax(p_subtypes,axis=1)==k,np.max(p_subtypes,axis=1)>0)
            idx_select[np.isnan(np.sum(p_subtypes,axis=1))] = 0
            p_yes = np.zeros((np.sum(idx_select),N))
            for i in range(N):

                wlikeli_norm = (1-mixing[i,k])*stats.norm.pdf(data_corrected[idx_select,i], 
                    loc = self.controls[i]['mu'][0], scale = self.controls[i]['std'][0])
                wlikeli_abnorm = (mixing[i,k])*stats.norm.pdf(data_corrected[idx_select,i], 
                    loc = self.cases[i]['mu'][0,k], scale = self.cases[i]['std'][0,k])

                p_yes[:,i] = np.divide(wlikeli_abnorm , (wlikeli_abnorm + wlikeli_norm))
            p_yes_list.append(p_yes)
            
        return p_yes_list
