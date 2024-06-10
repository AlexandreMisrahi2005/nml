# EE-452 Network Machine Learning — Course Final Project
### QM9 dataset for molecular graph property prediction

It is recommended to create a virtual environment to replicate the experiments. 

For example with conda:

```conda create -n "nml" python=3.11.5```

... activate it ...

```conda activate nml```

... and install the required packages.

```pip install -r requirements.txt```

Note that one may have to install the dependencies for torch-geometric separately depending on the machine used and if conda is available. 

Disclaimer: All files and folders ending with '_pamnet' contain code taken or modified from the [PAMNet repository](https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN).

Explanation of other files and folders:
- ```run.py```: main file for running experiments. Here you can set hyperparameters, choose the model, regression target, etc.
- ```run_extra_features.py```: similar to run.py but runs an experiment comparing all extra features.
- ```visualizations.ipynb```: notebook for generating the graphs used in the report.
- ```utils.py```: utility classes and functions used for pytorch-geometric transforms applied to the QM9 dataset.
- ```trainers.py```: training & evaluation functions for each model + some utility functions.
- ```models.py```: definition of the baseline models.
- ```predictions/```: logging files and loss histories saved to npy format.
- ```figures/```: figures used in the report.