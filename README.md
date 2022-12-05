# snowphlake
Staging NeurOdegeneration With PHenotype informed progression timeLine of biomarKErs

### After cloning, install using pip install -e ./

import snowphlake as spl

### A typical call is shown here:

T = spl.timeline(estimate_uncertainty=False, estimate_subtypes = True,
    subtyping_measure = 'zscore',\
    diagnostic_labels=['CN', 'SCD', 'MCI', 'AD'], n_maxsubtypes=6,\
    random_seed=100, n_nmfruns=50000, n_cpucores = 50)

S, Sboot = T.estimate(data,diagnosis,biomarkers_selected)

T.sequence_model['ordering'] contains all the predicted orderings of biomarkers
T.n_optsubtypes contains the optimum number of subtypes selected
S contains all patient-specific information

Once trained, spl.predict function can be used for the test-set
