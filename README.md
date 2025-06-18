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
The input format for data, diagnosis, and biomarkers_selected are explained below:

data: A NxM numpy matrix with no missing values. N = the number of patients in the training dataset. M = number of biomarkers for each patient. Each row must correspond to one timepoint for a patient. Please note that the method does not account for multiple timepoints per patient.
diagnosis: N x 1 numpy matrix with no missing values. Each element in this array should be a string that correspond to one of the elements in the diagnostic\_labels variable defined above. diagnostic\_labels should in turn be mentioned in the order of increasing severity of clinical staging.
biomarkers\_selected: A list of length M. Each element in the list should correspond to the name of the biomarker used, with no special characters.

T.sequence_model['ordering'] contains all the predicted orderings of biomarkers
T.n_optsubtypes contains the optimum number of subtypes selected
S contains all patient-specific information

Once trained, spl.predict function can be used for the test-set.
