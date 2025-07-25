### Ensure both python and R are installed
This can be done by:
conda create -c conda-forge --name snowphlake python=3.10 R=4.1

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
### After cloning this repository, install using pip install -e ./
Check how to use pip within a conda environment: https://stackoverflow.com/a/43729857 

After installation, this should work: import snowphlake as spl