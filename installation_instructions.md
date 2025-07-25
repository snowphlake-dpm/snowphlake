### Ensure both python and R are installed
This can be done by:
conda create -c conda-forge --name snowphlake python=3.10 R=4.1 ipython

### Install necessary packages in R
```
conda init
source activate snowphlake
R
```

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
### Final step
```
pip install git+https://github.com/snowphlake-dpm/snowphlake.git
```
