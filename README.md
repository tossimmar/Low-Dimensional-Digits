# Introduction

We present a derivation and implementation of a supervised dimensionality reduction and classification algorithm that is related (if not identical) to Kernel Fisher Discriminant Analysis (KFDA). The algorithm is applied to the [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset to visualize 8x8 pixel images of hand-written digits as points in a 1, 2, or 3-dimensional space. 

# Table of Contents

- [Provided Files](#provided-files)
- [Command-Line Interface](#command-line-interface)
- [Usage](#usage)
- [Expected Output](#expected-output)

# Provided Files

- **derivation (WIP).pdf**: Derivation of a supervised dimensionality reduction and classification algorithm that is related (if not identical) to KFDA. The provided **util.py** and **kfda.py** files are a direct implementation.

- **util.py**: Implementation of the algorithm's underlying operations.

- **kfda.py**: Representation of the algorithm.

- **optim.py**: Bayesian hyper-parameter optimization objective function implementation. 

- **scatter.py**: 1, 2, and 3-dimensional scatter plot functionality.

- **cli.py**: Command-line interface. 

- **main.py**: Application of the algorithm to the [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset to visualize 8x8 pixel images of hand-written digits as points in a 1, 2, or 3-dimensional space.

- **environment.yml**: Specifications and dependencies required to set up an environment for this project.

# Command-Line Interface

Various settings and hyper-parameter ranges can be specified via a command-line interface. The hyper-parameter ranges are used to tune the algorithm's corresponding hyper-parameters via Bayesian optimization, where the objective function is average $k$-fold cross-validation classification accuracy. The command-line interface is as follows:

- `-ts` `--train_size`: Proportion of data to allocate to training. For example, `-ts 0.75` specifies that 75% of the data should be allocated to training and 25% to testing. Default is 2/3.
  
- `-kf` `--k_folds`: Number of folds in $k$-fold cross-validation. For example, `-kf 5` specifies that the training data will be split into 5 folds. Default is 5.

- `-od` `--output_dim`: Dimension of low-dimensional representations. For example, `-od 3` specifies 3-dimensional representations.

- `-cd` `--centre_data`: If True, low-dimensional representations are centred at the origin. For example, `-cd True` specifies that low-dimensional representations are centred at the origin. Default is True.
  
-  `-r` `--regularization`: Type of regularization to apply. There are two choices: *coefficients* and *functional*. For example, `-r coefficients` specifies that large functional coefficients should be penalized, as discussed in the provided **derivation (WIP).pdf** file.

-  `-rr` `--reg_param_range`: Lower and upper bounds on the regularization hyper-parameter. For example, `-rr 1e-6 1` specifies that the regularization hyper-parameter search space is the closed interval $[0.000001, 1]$.
  
-  `-k` `--kernel`: Kernel function. There are seven choices: *chi2*, *linear*, *polynomial*, *rbf*, *laplacian*, *sigmoid*, and *cosine*. For example, `-k rbf` specifies the RBF (Gaussian) kernel.
  
-  `-dr` `--degree_range`: Lower and upper bounds on the degree of the polynomial kernel (only applicable if `-k polynomial` has been specified). For example, `-dr 2 4` specifies that the degree hyper-parameter search space is the set $\\{2, 3, 4\\}$.
  
-  `-gr` `--gamma_range`: Lower and upper bounds on the polynomial, sigmoid, rbf, laplacian, and chi-squared kernels' gamma parameter. For example, `-gr 1e-4 1e-2` specifies that the gamma hyper-parameter search space is the closed interval $[0.0001, 0.01]$.
  
-  `-cr` `--coef0_range`: Lower and upper bounds on the polynomial and sigmoid kernels' coef0 parameter. For example, `-cr 0 1` specifies that the coef0 hyper-parameter search space is the closed interval $[0, 1]$.
  
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
2. **kfda-valid-plot.html**: An interactive visualization of the validation/testing data as points in a 1, 2, or 3-dimensional space. Each cluster of points is labelled with the digit they represent. White points are those points that were misclassified by the algorithm.
