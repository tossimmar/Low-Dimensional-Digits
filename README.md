# Table of Contents

- [Overview](#overview)
- [Provided Files](#provided-files)
- [Command-Line Interface](#command-line-interface)
- [Usage](#usage)
- [Expected Output](#expected-output)

# Overview

We present a derivation and implementation of a supervised dimensionality reduction and classification algorithm that is related (if not identical) to kernel Fisher discriminant analysis. The algorithm is applied to the [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset to visualize 8x8 pixel images of hand-written digits as points in a 1, 2, or 3-dimensional space. 

The algorithm works on classification data. Feature vectors are mapped to a reproducing kernel Hilbert space where they are projected to low-dimensional representations such that representations with the same label are close together and representations with different labels are far apart. Distance is measured in terms of variance -- that is, the algorithm seeks to minimize "within-group" variance while maximizing "between-group" variance. Classification is achieved by applying any classifier (e.g., nearest centroid) to the low-dimensional data.

In our application of the algorithm to the Digits dataset, hyper-parameters are selected via Bayesian hyper-parameter optimization, where the objective function is average $k$-fold cross-validation classification accuracy. The optimal model (i.e., the model fit using the best hyper-parameters) is then used to generate 1, 2, or 3-dimensional representations of both the training and testing data. The low-dimensional data is then visualized in interactive plots. 

# Provided Files

- **derivation (WIP).pdf**: Derivation of a supervised dimensionality reduction and classification algorithm that is related (if not identical) to kernel Fisher discriminant analysis. The provided **util.py** and **kfda.py** files are a direct implementation.

- **util.py**: Implementation of the algorithm's underlying operations.

- **kfda.py**: Representation of the algorithm.

- **optim.py**: Bayesian hyper-parameter optimization objective function implementation. 

- **scatter.py**: 1, 2, and 3-dimensional scatter plot functionality.

- **cli.py**: Command-line interface. 

- **main.py**: Application of the algorithm to the [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset to visualize 8x8 pixel images of hand-written digits as points in a 1, 2, or 3-dimensional space.

- **environment.yml**: Specifications and dependencies required to set up an environment for this project.

# Command-Line Interface

- `-ts` `--train_size`: Proportion of data to allocate to training. For example, `-ts 0.75` specifies that 75% of the data should be allocated to training and 25% to testing.
  
- `-kf` `--k_folds`: Specifies the Bayesian hyper-parameter optimization objective function given by average $k$-fold cross-validation classification accuracy. For example, `-kf 5` specifies that the objective function is average $5$-fold cross-validation classification accuracy.

- `-od` `--output_dim`: Dimension of low-dimensional representations. For example, `-od 3` specifies 3-dimensional representations.

- `-cd` `--centre_data`: Whether low-dimensional data should be centred at the origin. For example, `-cd True` specifies that low-dimensional representations are centred at the origin.
  
-  `-r` `--regularization`: Type of regularization to apply. There are two choices: *coefficients* and *functional*. For example, `-r coefficients` specifies that large functional coefficients should be penalized, as discussed in the provided **derivation (WIP).pdf** file.

-  `-rr` `--reg_param_range`: Lower and upper bounds on the regularization hyper-parameter. For example, `-rr 1e-6 1` specifies that the regularization hyper-parameter search space is the closed interval $[0.000001, 1]$.
  
-  `-k` `--kernel`: Kernel function. There are seven choices: *chi2*, *linear*, *polynomial*, *rbf*, *laplacian*, *sigmoid*, and *cosine*. For example, `-k rbf` specifies the RBF (Gaussian) kernel.
  
-  `-dr` `--degree_range`: Lower and upper bounds on the polynomial kernel degree hyper-parameter. For example, `-dr 2 4` specifies that the degree hyper-parameter search space is the set $\\{2, 3, 4\\}$.
  
-  `-gr` `--gamma_range`: Lower and upper bounds on the polynomial, sigmoid, rbf, laplacian, and chi-squared kernels' gamma hyper-parameter. For example, `-gr 1e-4 1e-2` specifies that the gamma hyper-parameter search space is the closed interval $[0.0001, 0.01]$.
  
-  `-cr` `--coef0_range`: Lower and upper bounds on the polynomial and sigmoid kernels' coef0 hyper-parameter. For example, `-cr 0 1` specifies that the coef0 hyper-parameter search space is the closed interval $[0, 1]$.
  
-  `-nt` `--n_trials`: Number of trials in Bayesian hyper-parameter optimization. For example, `-nt 20` specifies that the objective function will be evaluated 20 times in an attempt to maximize it. 

# Usage

1. Install Miniconda or Anaconda if you haven't already.
2. Download or clone the repository.
3. Navigate to the project directory.
4. Create a new conda environment using the provided **environment.yml** file:
   
   &nbsp;&nbsp;&nbsp;&nbsp;`conda env create -f environment.yml`
   
   This command will create a new virtual environment with the necessary dependencies.
5. Activate the newly created environment:
   
   &nbsp;&nbsp;&nbsp;&nbsp;`conda activate kfda_env`

7. Run the following command:

   &nbsp;&nbsp;&nbsp;&nbsp;`python main.py [options...]`

   Replace `[options...]` with any of the available command-line options provided by the command-line interface. For example,

   &nbsp;&nbsp;&nbsp;&nbsp;`python main.py -od 3 -r functional -rr 1e-6 1 -k rbf -gr 1e-4 1e-2 -nt 20`

# Expected Output

### Terminal Output
During the execution of **main.py**, the terminal will display details of the Bayesian hyper-parameter optimization process. 

### HTML Files
1. **kfda-train-plot.html**: An interactive visualization of the training data as points in a 1, 2, or 3-dimensional space. Each cluster of points is labelled with the digit they represent.
2. **kfda-test-plot.html**: An interactive visualization of the testing data as points in a 1, 2, or 3-dimensional space. Each cluster of points is labelled with the digit they represent. White points are those points that were misclassified by the algorithm.
