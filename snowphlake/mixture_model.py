# Author: Vikram Venkatraghavan, Amsterdam UMC

import numpy as np
import pandas as pd
import sklearn
import scipy as sp 
from scipy import stats
from scipy import optimize 
import multiprocessing

# Need scipy 1.7.1 or higher

class dirichlet_process():

    def __init__(self, N, biomarker_labels, n_gaussians=1, n_maxsubtypes=1,
                    random_seed=42, estimate_mixing="mcmc", niter_tunein= 4000, niter_trace=1000):
        # N is the number of events in the model.

        self.n_gaussians = n_gaussians
        self.n_maxsubtypes = n_maxsubtypes
        self.random_seed = random_seed
        if n_maxsubtypes==1:
            self.estimate_mixing = estimate_mixing # "mcmc" or "debm-2019"
        else:
            self.estimate_mixing = "mcmc"
        self.biomarker_labels = biomarker_labels
        self.niter_tunein = niter_tunein
        self.niter_trace = niter_trace

        self.controls = [{"mu":np.zeros(self.n_gaussians) ,"std":np.zeros(self.n_gaussians), 
                        "weights": np.zeros(self.n_gaussians)+1/n_gaussians} for x in range(N)]
        self.cases = [{"mu":np.zeros((self.n_gaussians,self.n_maxsubtypes)),
                    "std":np.zeros((self.n_gaussians,self.n_maxsubtypes)), 
                        "weights": np.zeros((self.n_gaussians,self.n_maxsubtypes)) + 1/n_gaussians} for x in range(N)] 
        self.mixing = np.zeros((N,self.n_maxsubtypes)) + 0.5
        self.DPmm_controls = None # for sklearn DPGMM when n_gaussians > 1
        self.DP_subtyping = {"model":None, "trace":None} # for pymc3 Dirichlet process mixutures when n_maxsubtypes > 1
    
    def fit(self, data_corrected, diagnosis):
        
        ## First implementation for n_gaussians=1 and n_maxsubtypes=1
        # Truncated Gaussian for initialization 
        N = data_corrected.shape[1]

        def calculate_total_likelihood(data, n_gaussians, n_subtypes, controls, cases, mixing):
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
            
            return total_likeli

        def objective_mixing(mixing,data,cases,controls):
            n_gaussians = controls['mu'].shape[0]
            n_subtypes = cases['mu'].shape[1]

            total_likeli = calculate_total_likelihood(data, n_gaussians, n_subtypes, 
                                        controls, cases, mixing)
            objfun = -np.sum(np.log(total_likeli))

            return objfun
        
        def objective_cases_distribution(cases, data, controls, mixing):

            n_gaussians = controls['mu'].shape[0]
            n_subtypes = int(cases.shape[0] / (3*n_gaussians)) ## Check this

            total_likeli = calculate_total_likelihood(data, n_gaussians, n_subtypes, 
                                        controls, cases, mixing)
            objfun = -np.sum(np.nanmean(np.log(total_likeli),axis=0))

            return objfun 
        
        def debm2019(self, data_corrected, diagnosis):
            
            N = data_corrected.shape[1]
            idx_cn = diagnosis == 1 
            idx_cases = diagnosis == np.nanmax(diagnosis)
            flag_opt_stop=0
            cnt = 0
            while flag_opt_stop==0:
                cnt = cnt+1
                mixing0 = np.copy(self.mixing)
                if cnt==1:
                    mixing_init = np.copy(mixing0)
                # Original DEBM implementation
                for i in range(N):
                    bnd_mixing = np.asarray([0.05,0.95])
                    bnd_mixing = np.repeat(bnd_mixing[np.newaxis,:],self.n_maxsubtypes,axis=0)
                    res=optimize.minimize(objective_mixing,mixing0[i],
                                    args=(data_corrected[~idx_cn,i],self.cases[i],self.controls[i]),
                                    method='SLSQP', bounds=bnd_mixing)
                    self.mixing[i,0] = res.x

                
                if np.mean(np.abs(self.mixing[:,0]-mixing0[:,0])) < 0.01: 
                    flag_opt_stop=1
                    break 

                ## Check for diverging mixing parameter in small datasets  to introduce sanity check

                for i in range(N):

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

        def mcmc(self, data_corrected, diagnosis):
            
            import pymc3 as pm 
            from theano import tensor as tt

            idx_cn = diagnosis==1
            idx_cases = diagnosis==np.nanmax(diagnosis)

            N = data_corrected.shape[1]
            self.DP_subtyping["model"] = pm.Model()

            if self.n_maxsubtypes > 1:
                with self.DP_subtyping["model"]:
                    alphaS = pm.Gamma("alphaS", 1.0, 1.0)
                    subtypes = pm.Dirichlet("subtypes", a=np.zeros(self.n_maxsubtypes)+alphaS, shape=self.n_maxsubtypes)
                    mixing = pm.Uniform("mixing", 0.05, 0.95, shape=(N,self.n_maxsubtypes))
                    comp_logps = []
                    for kk in range(self.n_maxsubtypes):
                        comp_logps_row = []
                        for i in range(N):
                            lower_bound = np.min([np.nanmin(data_corrected[idx_cases,i]),
                                            self.controls[i]['mu'][0]])
                            upper_bound = np.max([np.nanmax(data_corrected[idx_cases,i]),
                                            self.controls[i]['mu'][0]])
                            muA = pm.Uniform("muA_"+self.biomarker_labels[i]+'_subtype'+str(kk), 
                                    lower_bound, upper_bound)
                            stdA = pm.Uniform("stdA_"+self.biomarker_labels[i]+'_subtype'+str(kk),
                                    self.cases[i]['std'][0,0],np.std(data_corrected[~idx_cn,i]))
                            ind_logps = pm.NormalMixture.dist(tt.stack([mixing[i,kk],1-mixing[i,kk]]),
                                mu = tt.stack([muA,self.controls[i]['mu'][0]]),
                                sd = tt.stack([stdA,self.controls[i]['std'][0]])).logp(data_corrected[~idx_cn,i])
                            comp_logps_row.append(ind_logps.sum())
                        comp_logps.append(comp_logps_row)
                    total_logp = pm.Potential('subtype_mixture', obj_subtyping(subtypes, comp_logps, mixing))
            else:
                with self.DP_subtyping["model"]:
                    total_logp = 0
                    mixing = pm.Uniform("mixing", 0.05, 0.95, shape=N)
                    for i in range(N):
                        diff_mu = np.abs(self.cases[i]['mu'][0,0] - self.controls[i]['mu'][0])
                        muA = pm.Normal("muA_"+self.biomarker_labels[i],
                                mu=self.cases[i]['mu'][0,0],sd = diff_mu/3)
                        stdA = pm.Uniform("stdA_"+self.biomarker_labels[i],
                                    self.cases[i]['std'][0,0],np.std(data_corrected[~idx_cn,i]))
                        ind_logp = pm.NormalMixture.dist(tt.stack([mixing[i],1-mixing[i]]), 
                                mu = tt.stack([muA,self.controls[i]['mu'][0]]),
                                sd = tt.stack([stdA,self.controls[i]['std'][0]])).logp(data_corrected[~idx_cn,i])
                        total_logp = total_logp - pm.Potential("logp_"+self.biomarker_labels[i], ind_logp.sum())
                
            with self.DP_subtyping["model"]:

                self.DP_subtyping["trace"] = pm.sample(self.niter_trace, tune=self.niter_tunein, chains=2, 
                    cores=2*multiprocessing.cpu_count(), init="advi", target_accept=0.9,
                    random_seed=self.random_seed, return_inferencedata=False)
                
            if self.n_maxsubtypes>1:
                self.mixing[:,:] = self.DP_subtyping["trace"]["mixing"].mean(axis=0)
                for i in range(N):
                    for kk in range(self.n_maxsubtypes):
                        self.cases[i]['std'][0,kk] = \
                            self.DP_subtyping["trace"]["stdA_"+self.biomarker_labels[i]+'_subtype'+str(kk)].mean(axis=0)
                        self.cases[i]['mu'][0,kk] = \
                            self.DP_subtyping["trace"]["muA_"+self.biomarker_labels[i]+'_subtype'+str(kk)].mean(axis=0)
            else:
                self.mixing[:,0] = self.DP_subtyping["trace"]["mixing"].mean(axis=0)
                for i in range(N):
                    self.cases[i]['std'][0,0] = \
                        self.DP_subtyping["trace"]["stdA_"+self.biomarker_labels[i]].mean(axis=0)
                    self.cases[i]['mu'][0,0] = \
                        self.DP_subtyping["trace"]["muA_"+self.biomarker_labels[i]].mean(axis=0)
            
            return 

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
        
        # Optimization
        if self.estimate_mixing=='mcmc':
            mcmc(self, data_corrected, diagnosis)
        else:
            debm2019(self, data_corrected, diagnosis)
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
