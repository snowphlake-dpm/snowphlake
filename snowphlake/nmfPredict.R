# Code by Vikram Venkatraghavan
# adapted from Nicolas Sauwen's code (maintainer of the R NMF package)

library(nnls)
library(NMF)

# NMFResult should be the result of a single NMF call, 
# ideally after the model selection step. For example: 
# NMFResult <- nmf(D,5,method='nsNMF', seed=123456, nrun=30)

# DTest is the test dataset

# This predict functionality works for nsNMF option. 

predictNMF <- function(DTest, basis, theta) {

rank <- ncol(basis)

S <- (1-theta)*diag(rank) + (theta / rank) * rep(1,rank)%*%t(rep(1,rank))
WS <- basis%*%S ## For the Non-smooth NMF version

coeff <- matrix(0, rank, ncol(DTest))
for(i in 1:ncol(DTest)) { # This loops over each subject in the test dataset
    nlsFit <- nnls::nnls(WS,DTest[,i]) 
    coeff[,i] <- stats::coefficients(nlsFit)
}

predictResult <- NMF::nmfModel(W = WS, H = coeff)
return(predictResult)
}