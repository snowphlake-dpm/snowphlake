# Snowphlake - A tool for identifying disease subtypes and modeling disease progression
Snowphlake stands for: Staging NeurOdegeneration With PHenotype informed progression timeLine of biomarKErs

If you are using this work, please cite our paper: [A large-scale multi-centre study characterising atrophy heterogeneity in Alzheimer's disease](https://doi.org/10.1016/j.neuroimage.2025.121381)

![Subtypes in AD](./img.jpg)

## Installation Instructions

Click [here](./installation_instructions.md) for installation instructions

### A typical call on a toy dataset is shown here:
```
import numpy as np
import snowphlake as spl
from snowphlake.load_dataset import load_dataset
from sklearn.impute import SimpleImputer

data=load_dataset() #Toy dataset
diagnosis=data['Diagnosis'].values
data=data.drop(['Diagnosis','PTID','Age','Sex','ICV','EXAMDATE'],axis=1)
#Remove the effects of the confounders before calling snowphlake. This is not done in the toy dataset

#Impute missing values if necessary.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data.values)
data_imputed=imp.transform(data.values)
biomarkers_selected = list(data)

T = spl.timeline(estimate_uncertainty=False, estimate_subtypes = True,\
    diagnostic_labels=['CN', 'MCI', 'AD'], n_maxsubtypes=6, model_selection='full',\
    random_seed=100, n_nmfruns=8550, n_cpucores = 1)

S, Sboot = T.estimate(data_imputed,diagnosis,biomarkers_selected)
```
"n_nmfruns" should roughly be equal to $25 x n_{AD}$, where $n_{AD}$ is the number of patients with highest clinical stage (e.g. with AD dementia).

If "estimate_subtypes" is False, the code estimates a progression timeline (without subtyping) using [DEBM](https://doi.org/10.1016/j.neuroimage.2018.11.024)

If "n_cpucores" $>1$, subtyping is done using parallel processing.

The input format for data, diagnosis, and biomarkers_selected are explained below:

**data**: A $N x M$ numpy matrix with no missing values. $N$ = the number of patients in the training dataset. $M$ = number of biomarkers for each patient. Each row must correspond to one timepoint for a patient. Please note that the method does not account for multiple timepoints per patient.

**diagnosis**: $N x 1$ numpy matrix with no missing values. Each element in this array should be a string that correspond to one of the elements in the "diagnostic\_labels" variable defined above. "diagnostic\_labels" should in turn be mentioned in the order of increasing severity of clinical staging.

**biomarkers\_selected**: A list of length $M$. Each element in the list should correspond to the name of the biomarker used, with no special characters.

T.sequence_model['ordering'] contains all the predicted orderings of biomarkers
T.n_optsubtypes contains the optimum number of subtypes selected
S contains all patient-specific information

Once trained, spl.predict function can be used for the test-set.
