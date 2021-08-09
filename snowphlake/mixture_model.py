# Author: Vikram Venkatraghavan, Amsterdam UMC

import numpy as np
import pandas as pd
import sklearn
import scipy as sp 
from scipy import stats
from scipy import optimize 

# Need scipy 1.7.1 or higher

class dirichlet_process():

    def __init__(self, N, n_gaussians=1, n_maxsubtypes=1, random_seed=42):
        # N is the number of events in the model.

        self.n_gaussians = n_gaussians
        self.n_maxsubtypes = n_maxsubtypes
        self.random_seed = random_seed

        self.controls = [{"mu":np.zeros(self.n_gaussians) ,"std":np.zeros(self.n_gaussians), 
                        "weights": np.zeros(self.n_gaussians)+1/n_gaussians} for x in range(N)]
        self.cases = [{"mu":np.zeros((self.n_gaussians,self.n_maxsubtypes)),
                    "std":np.zeros((self.n_gaussians,self.n_maxsubtypes)), 
                        "weights": np.zeros((self.n_gaussians,self.n_maxsubtypes)) + 1/n_gaussians} for x in range(N)] 
        self.mixing = np.zeros((N,self.n_maxsubtypes)) + 0.5
    
    def fit(self, data_corrected, diagnosis):
        
        ## First implementation for n_gaussians=1 and n_maxsubtypes=1
        # Truncated Gaussian for initialization 
        N = data_corrected.shape[1]

        def objective_mixing(mixing,data,cases,controls):
            n_gaussians = controls['mu'].shape[0]
            n_subtypes = cases['mu'].shape[1]

            total_likeli = np.zeros(data.shape[0])
            likeli_norm = np.zeros(data.shape[0])
            likeli_abnorm = np.zeros((data.shape[0],n_subtypes))
            for j in range(n_gaussians):
                dist_norm=stats.norm(loc = controls['mu'][j], 
                                    scale = controls['std'][j])
                ## This weighted addition for multiple Gaussians is questionable
                likeli_norm[:] = likeli_norm[:] + (dist_norm.pdf(data) * controls['weights'][j])
                for k in range(n_subtypes):
                    dist_abnorm = stats.norm(loc = cases['mu'][j,k], scale = cases['std'][j,k])
                    likeli_abnorm[:,k] = likeli_abnorm[:,k] + (dist_abnorm.pdf(data) * cases['weights'][j,k])

            for k in range(n_subtypes):
                ## This is now a simple addition. 
                ## Change this later to include probabilities from the dirichlet process
                total_likeli[:] = total_likeli[:] + (likeli_abnorm[:,k]*mixing[k]) + (likeli_norm[:] * (1-mixing[k]))
            
            objfun = -np.sum(np.log(total_likeli))

            return objfun
        
        def objective_cases_distribution(cases, data, controls, mixing):

            n_gaussians = controls['mu'].shape[0]
            n_subtypes = int(cases.shape[0] / (3*n_gaussians)) ## Check this

            total_likeli = np.zeros(data.shape[0])
            likeli_norm = np.zeros(data.shape[0])
            likeli_abnorm = np.zeros((data.shape[0],n_subtypes))
            for j in range(n_gaussians):
                dist_norm=stats.norm(loc = controls['mu'][j], 
                                    scale = controls['std'][j])
                ## This weighted addition for multiple Gaussians is questionable
                likeli_norm[:] = likeli_norm[:] + (dist_norm.pdf(data) * controls['weights'][j])
                for k in range(n_subtypes): ## Works only for k == 1 
                    dist_abnorm = stats.norm(loc = cases[0], scale = cases[1])
                    likeli_abnorm[:,k] = likeli_abnorm[:,k] + (dist_abnorm.pdf(data) * 1)
                    ## Replace this with cases weights later            
            
            for k in range(n_subtypes):
                ## This is now a simple addition. 
                ## Change this later to include probabilities from the dirichlet process
                total_likeli[:] = total_likeli[:] + (likeli_abnorm[:,k]*mixing[k]) + (likeli_norm[:] * (1-mixing[k]))
            
            objfun = -np.sum(np.log(total_likeli))

            return objfun 

        idx_cn = diagnosis == 1 
        idx_cases = diagnosis == np.nanmax(diagnosis)
        for i in range(N):
            # Initial Fit
            self.controls[i]['mu'][0],self.controls[i]['std'][0]=stats.norm.fit(data_corrected[idx_cn,i])
            self.cases[i]['mu'][0,0], self.cases[i]['std'][0,0]=stats.norm.fit(data_corrected[idx_cases,i])
            
            # Reject overlapping regions 
            likeli_norm = stats.norm.pdf(data_corrected[idx_cases,i], 
                loc = self.controls[i]['mu'][0], scale = self.controls[i]['std'][0])
            likeli_abnorm = stats.norm.pdf(data_corrected[idx_cases,i], 
                loc = self.cases[i]['mu'][0,0], scale = self.cases[i]['std'][0,0])
            idx_reject = likeli_norm > likeli_abnorm

            # Truncated Fit --> make this for multiple gaussians
            self.cases[i]['mu'][0,0], self.cases[i]['std'][0,0]= \
                stats.norm.fit(data_corrected[idx_cases,i][~idx_reject])
        
            # Alternating optimization
            flag_opt_stop=0
            cnt = 0
            while flag_opt_stop==0:
                cnt = cnt+1 
                mixing0 = np.copy(self.mixing[i,:])
                if cnt==1:
                    mixing_init = mixing0 

                bnd_mixing = np.asarray([0.05,0.95])
                bnd_mixing = np.repeat(bnd_mixing[np.newaxis,:],self.n_maxsubtypes,axis=0)
                res=optimize.minimize(objective_mixing,mixing0,
                            args=(data_corrected[~idx_cn,i],self.cases[i],self.controls[i]),
                            method='SLSQP', bounds=bnd_mixing)
                self.mixing[i,:] = res.x

                if np.abs(res.x-mixing0)[0] < 0.01:
                    flag_opt_stop=1
                    break 

                ## Check for diverging mixing parameter in small datasets  to introduce sanity check
                
                cases0 = np.asarray([self.cases[i]['mu'].flatten(), self.cases[i]['std'].flatten()])
                ## Excluding weights for now in this optimization
                ## Check the order of flattening when introducing subtypes 
                ## Also when introducing multiple Gaussians
                cases0 = np.asarray([self.cases[i]['mu'].flatten(), self.cases[i]['std'].flatten()])
                ## ,                        self.cases[i]['weights'].flatten()])

                bnd_cases = np.zeros((2,2))
                if np.mean(self.cases[i]['mu'][:])<np.mean(self.controls[i]['mu'][:]):
                    bnd_cases[0,:] = np.asarray([np.nanmin(data_corrected[idx_cases,i]),
                                                self.controls[i]['mu'][0]])
                    
                else:
                    bnd_cases[0,:] = np.asarray([self.controls[i]['mu'][0],
                                                np.nanmax(data_corrected[idx_cases,i])])

                bnd_cases[1,:] = np.asarray([0,np.std(data_corrected[idx_cases,i])])

                res=optimize.minimize(objective_cases_distribution,cases0,
                            args=(data_corrected[~idx_cn,i],self.controls[i],self.mixing),
                            method='SLSQP',bounds=bnd_cases)
                
                ## Check this too while introducing subtypes and multiple Gaussians
                self.cases[i]['mu'][0,0]=res.x[0]
                self.cases[i]['std'][0,0]=res.x[1]
                ##self.cases[i]['weights'][0,0]=res.x[2]

                # --> do a sanity check for rejecting the optimization
                # if GMM fails, revert to initialized values

        return 
    
    def predict_posterior(self,data_corrected):
        
        N = data_corrected.shape[1]
        p_yes = np.zeros((data_corrected.shape[0],N))

        for i in range(N):

            wlikeli_norm = (1-self.mixing[i])*stats.norm.pdf(data_corrected[:,i], 
                loc = self.controls[i]['mu'][0], scale = self.controls[i]['std'][0])
            wlikeli_abnorm = (self.mixing[i])*stats.norm.pdf(data_corrected[:,i], 
                loc = self.cases[i]['mu'][0,0], scale = self.cases[i]['std'][0,0])

            p_yes[:,i] = np.divide(wlikeli_abnorm , (wlikeli_abnorm + wlikeli_norm))
            
        return p_yes
