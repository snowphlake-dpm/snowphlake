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
                    random_seed=42, estimate_mixing="mcmc", niter_tunein= 4000,
                    niter_trace=1000, optim_thresh = 0.05):
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
        self.optim_thresh = optim_thresh

        self.controls = [{"mu":np.zeros(self.n_gaussians) ,"std":np.zeros(self.n_gaussians), 
                        "weights": np.zeros(self.n_gaussians)+1/n_gaussians} for x in range(N)]
        self.cases = [{"mu":np.zeros((self.n_gaussians,self.n_maxsubtypes)),
                    "std":np.zeros((self.n_gaussians,self.n_maxsubtypes)), 
                        "weights": np.zeros((self.n_gaussians,self.n_maxsubtypes)) + 1/n_gaussians} for x in range(N)] 
        self.cases_init = [{"mu":np.zeros((self.n_gaussians,self.n_maxsubtypes)),
                    "std":np.zeros((self.n_gaussians,self.n_maxsubtypes)), 
                        "weights": np.zeros((self.n_gaussians,self.n_maxsubtypes)) + 1/n_gaussians} for x in range(N)] 
        self.mixing = np.zeros((N,self.n_maxsubtypes)) + 0.5
        self.DP_controls = [{"model":None, "trace":None, "biomarker": None} for x in range(N)]
        self.DP_subtyping = {"model":None, "trace":None} # for pymc3 Dirichlet process mixutures when n_maxsubtypes > 1
        self.DP_cases = [[{"model":None, "trace":None, "biomarker": None} for x in range(N)] \
                                for y in range(self.n_maxsubtypes)]
    
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
        
        def debm2019(self, data_corrected, idx_cn):
            
            N = data_corrected.shape[1]
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

                
                if np.mean(np.abs(self.mixing[:,0]-mixing0[:,0])) < self.optim_thresh: 
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

        def obj_subtyping(subtypes, comp_logps, mixing):
            import pymc3 as pm 
            from theano import tensor as tt
            n_maxsubtypes = len(comp_logps)
            comp_logp_k=[]
            for k in range(n_maxsubtypes):
                total_comp_logp = 0
                N = len(comp_logps[k])
                for i in range(N):
                    total_comp_logp = total_comp_logp + (1)*(comp_logps[k][i])
                comp_logp_k.append(total_comp_logp)
            comp_logp = tt.stack(comp_logp_k)
            return pm.math.logsumexp(tt.log(subtypes) + comp_logp, axis=-1)

        def subtyping_model_init(self,data_corrected,idx_cn):
            import pymc3 as pm 
            from theano import tensor as tt

            self.DP_subtyping["model"] = pm.Model()
            if self.n_maxsubtypes > 1:
                with self.DP_subtyping["model"]:
                    alphaS = pm.Gamma("alphaS", 1.0, 1.0)
                    subtypes = pm.Dirichlet("subtypes", a=np.zeros(self.n_maxsubtypes)+alphaS, shape=self.n_maxsubtypes)
                    mixing = pm.Uniform("mixing", 0.05, 0.95, shape=(N,self.n_maxsubtypes))
                    comp_logps = []
                    for k in range(self.n_maxsubtypes):
                        comp_logps_row = []
                        for i in range(N):
                            ind_logps = pm.NormalMixture.dist(tt.stack([mixing[i,k],1-mixing[i,k]]),
                                    mu = tt.stack([self.cases[i]['mu'][0,k],self.controls[i]['mu'][0]]),
                                    sd = tt.stack([self.cases[i]['std'][0,k],self.controls[i]['std'][0]])).logp(data_corrected[~idx_cn,i])
                            comp_logps_row.append(ind_logps.sum())
                        comp_logps.append(comp_logps_row)
                    total_logp = pm.Potential('subtype_mixture', obj_subtyping(subtypes, comp_logps, mixing))
            else:
                with self.DP_subtyping["model"]:
                    total_logp = 0
                    mixing = pm.Uniform("mixing", 0.05, 0.95, shape=N)
                    for i in range(N):
                        ind_logp = pm.NormalMixture.dist(tt.stack([mixing[i],1-mixing[i]]), 
                                    mu = tt.stack([self.cases[i]['mu'][0,0],self.controls[i]['mu'][0]]),
                                    sd = tt.stack([self.cases[i]['std'][0,0],self.controls[i]['std'][0]])).logp(data_corrected[~idx_cn,i])
                        total_logp = total_logp - pm.Potential("logp_"+self.biomarker_labels[i], ind_logp.sum())

            return

        def cases_model_init(self, data_corrected, idx_cn): 
            
            import pymc3 as pm 
            from theano import tensor as tt
            ## this works only for n_gaussians = 1 
            for i in range(N):
                for k in range(self.n_maxsubtypes):
                    self.DP_cases[k][i]["model"] = pm.Model() 
                    with self.DP_cases[k][i]["model"]:
                        diff_mu = np.abs(self.cases_init[i]['mu'][0,k] - self.controls[i]['mu'][0])
                        muA = pm.Normal("mu_", 
                                mu=self.cases_init[i]['mu'][0,k], sd = diff_mu/3)
                        stdA = pm.Uniform("std_",
                                0,np.std(data_corrected[~idx_cn,i]))     
                        dist = pm.NormalMixture("dist",tt.stack([self.mixing[i,k],1-self.mixing[i,k]]), 
                                    mu = tt.stack([muA,self.controls[i]['mu'][0]]),
                                    sd = tt.stack([stdA,self.controls[i]['std'][0]]), observed = data_corrected[~idx_cn,i])

        def mcmc(self, data_corrected, idx_cn):
            
            import pymc3 as pm 
            from theano import tensor as tt

            N = data_corrected.shape[1]
            flag_opt_stop=0
            cnt =0 
            while flag_opt_stop==0:
                cnt = cnt+1
                mixing0 = np.copy(self.mixing)
                if cnt==1:
                    mixing_init = np.copy(mixing0)
                # Alternating mcmc optimization for subtyping+mixing, and distribution for cases 
                subtyping_model_init(self,data_corrected,idx_cn)
                with self.DP_subtyping["model"]:
                    self.DP_subtyping["trace"] = pm.sample(self.niter_trace, tune=self.niter_tunein, chains=2, 
                        cores=2*multiprocessing.cpu_count(), init="advi", target_accept=0.9,
                        random_seed=self.random_seed, return_inferencedata=False)
                
                if self.n_maxsubtypes==1:
                    self.mixing[:,0] = self.DP_subtyping["trace"]["mixing"].mean(axis=0)
                else:
                    self.mixing[:,:] = self.DP_subtyping["trace"]["mixing"].mean(axis=0)
                
                print ("mixing diff:", np.mean(np.abs(self.mixing[:,:]-mixing0[:,:])) )
                print (self.mixing)

                if np.mean(np.abs(self.mixing[:,:]-mixing0[:,:])) < self.optim_thresh:
                    flag_opt_stop=1
                    break

                cases_model_init(self, data_corrected, idx_cn) 
                for k in range(self.n_maxsubtypes):
                    for i in range(N):
                        print("Optimizing case distribution for biomarker:", self.biomarker_labels[i])
                        with self.DP_cases[k][i]["model"]:
                            self.DP_cases[k][i]["trace"] = pm.sample(self.niter_trace, tune=self.niter_tunein, chains=2, 
                            cores=2*multiprocessing.cpu_count(), init="advi", target_accept=0.9,
                            random_seed=self.random_seed, return_inferencedata=False)
                
                if self.n_maxsubtypes>1:
                    for i in range(N):
                        for k in range(self.n_maxsubtypes):
                            self.cases[i]['std'][0,k] = \
                                self.DP_cases[k][i]["trace"]["std_"].mean(axis=0)
                            self.cases[i]['mu'][0,k] = \
                                self.DP_cases[k][i]["trace"]["mu_"].mean(axis=0)
                else:
                    for i in range(N):
                        self.cases[i]['std'][0,0] = \
                            self.DP_cases[0][i]["trace"]["std_"].mean(axis=0)
                        self.cases[i]['mu'][0,0] = \
                            self.DP_cases[0][i]["trace"]["mu_"].mean(axis=0)
            
            return 

        idx_cn = diagnosis == 1 
        idx_cases = diagnosis == np.nanmax(diagnosis)
        for i in range(N):
            # Initial Fit
            self.controls[i]['mu'][0],self.controls[i]['std'][0]=stats.norm.fit(data_corrected[idx_cn,i])
            for k in range(self.n_maxsubtypes):
                self.cases[i]['mu'][0,k], self.cases[i]['std'][0,k]=stats.norm.fit(data_corrected[idx_cases,i])
            
            # Reject overlapping regions 
            likeli_norm = stats.norm.pdf(data_corrected[idx_cases,i], 
                loc = self.controls[i]['mu'][0], scale = self.controls[i]['std'][0])
            likeli_abnorm = stats.norm.pdf(data_corrected[idx_cases,i], 
                loc = self.cases[i]['mu'][0,0], scale = self.cases[i]['std'][0,0])
            idx_reject = likeli_norm > likeli_abnorm

            # Truncated Fit --> make this for multiple gaussians
            for k in range(self.n_maxsubtypes):
                self.cases[i]['mu'][0,k], self.cases[i]['std'][0,k]= \
                    stats.norm.fit(data_corrected[idx_cases,i][~idx_reject])
                self.cases_init[i]['mu'][0,k], self.cases_init[i]['std'][0,k]= \
                    stats.norm.fit(data_corrected[idx_cases,i][~idx_reject])
        
        # Optimization
        if self.estimate_mixing=='mcmc':
            mcmc(self, data_corrected, idx_cn)
        else:
            debm2019(self, data_corrected, idx_cn)
        return 
    
    def predict_posterior(self,data_corrected):
        
        N = data_corrected.shape[1]
        p_yes_list = []

        for k in range(self.n_maxsubtypes):
            p_yes = np.zeros((data_corrected.shape[0],N))

            for i in range(N):

                wlikeli_norm = (1-self.mixing[i,k])*stats.norm.pdf(data_corrected[:,i], 
                    loc = self.controls[i]['mu'][0], scale = self.controls[i]['std'][0])
                wlikeli_abnorm = (self.mixing[i,k])*stats.norm.pdf(data_corrected[:,i], 
                    loc = self.cases[i]['mu'][0,k], scale = self.cases[i]['std'][0,k])

                p_yes[:,i] = np.divide(wlikeli_abnorm , (wlikeli_abnorm + wlikeli_norm))
            p_yes_list.append(p_yes)
                
        return p_yes_list
