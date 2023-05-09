# snowphlake
Staging NeurOdegeneration With PHenotype informed progression timeLine of biomarKErs

### Ensure both python and R are installed
This can be done by:
conda create -c conda-forge --name snowphlake python=3.10 R=4.1

### Install necessary packages in R
conda init
source activate snowphlake
R

[1] Install Biobase in R:
```
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("Biobase")
```
[2] Install NMF and nnls in R:
```
install.packages('NMF')
install.packages('nnls')
```
### After cloning this repository, install using pip install -e ./
Check how to use pip within a conda environment: https://stackoverflow.com/a/43729857 

After installation, this should work: import snowphlake as spl

### A typical call is shown here:
```
T = spl.timeline(estimate_uncertainty=False, estimate_subtypes = True,
    subtyping_measure = 'zscore',\
    diagnostic_labels=['CN', 'SCD', 'MCI', 'AD'], n_maxsubtypes=6,\
    random_seed=100, n_nmfruns=50000, n_cpucores = 50)

S, Sboot = T.estimate(data,diagnosis,biomarkers_selected)
```
T.sequence_model['ordering'] contains all the predicted orderings of biomarkers
T.n_optsubtypes contains the optimum number of subtypes selected
S contains all patient-specific information

Once trained, spl.predict function can be used for the test-set
