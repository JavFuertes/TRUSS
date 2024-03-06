# Bayesian optimization for truss structures

Aim: Use Bayesian optimization on a truss optimization benchmark from literature.  

# Project Overview
This repository houses the artifacts and documentation related to our project, which focuses on Bayesian Optimization. The key components of this repository include a Jupyter notebook containing the Bayesian Optimization implementation, project planning document, and the final presentation materials.

### Repo structure and contents
- 口Reading includes much of the reference literature which was used as benchmark [Kanarachos](1-s2.0-S0045794916302036-main.pdf): the optimisation method which is being compared to a machine learning solution.
- **pyJive**: a finite element code that can compute the natural frequencies of a given truss design, which can be treated as black box model for this project
- **truss_bridge**: a directory with input files for the case of the project, including a [notebook](truss_bridge/truss_bridge.ipynb) with a demonstration of how to interact with the finite element code

The original paper and reference can be found here,
Investigation and reference results obtained with other optimization approaches, check this paper: [Kanarachos et al., 2017](https://dx.doi.org/10.1016/j.compstruc.2016.11.005)


## TRUSS notebook:
The Jupyter notebook contains the implementation of the Bayesian optimiser and the posterior analysis. It covers a from blank implementation, a solution through Botorch module and analysis on the covergence. 

## Presentation_final
A small breakdown of the progress made in the project [TrussBOPT.pptx].


### Useful links



Tutorials on Bayesian optimization:
- [https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec](https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec)
- [https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization](https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization)
