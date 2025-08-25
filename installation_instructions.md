### Ensure both python and R are installed
This can be done by:
conda env create --name snowphlake -f https://raw.githubusercontent.com/snowphlake-dpm/snowphlake/main/environment.yml

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
q()
```
### Final step
```
pip install --no-deps git+https://github.com/snowphlake-dpm/snowphlake.git
```
