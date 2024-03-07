# Bayesian optimization for truss structures

Structures that make optimal use of the material they are made of reduces the cost and environmental impact of their construction as the amount of material required. Optimization of structural design is a challenging task because of the high number of design parameters and the relatively expensive evaluation of the suitability of any given design. Standard optimization techniques in high-dimensional design space require a very large number of possible designs that need to be evaluated. In structural analysis, where evaluating the objective function and checking the constraints involves the solution of a structural mechanics problem, e.g. with finite elements, this quickly becomes very expensive, even if the model is relatively simple from structural point of view. Bayesian optimization is a machine-learning-based optimization technique that aims to reduce the number of evaluations of the objective function through data-driven exploration of the design space with a probabilistic surrogate.

<center>

| ![Multi-dimensional solution space](https://www.mathworks.com/help/examples/stats/win64/ParellelBayeianOptimizationExample_01.png) | ![TRUSS optimal solution](TRUSS1/TRUSS1/口Reading/Figures/Kanarachos/KNInitial.png) |
|----------------------------------------------------|----------------------------------------------------|
| **Figure 1**: Visualisation of a Multi-dimensional solution space | **Figure 2**: TRUSS optimisation problem formulation |
</center>


## Objective & Description:
---

<span style="font-size: larger;"><B>Project Objective:</B></span> To find the optimal truss design


To achieve this project objective, we need to find the optimal set of nodal coordinates and cross-sectional properties. This will allow us to minimize the total weight of the structure while satisfying various constraints related to the structure's natural frequencies. Finding solutions with low mass that also meet the natural frequency constraints demonstrates a methodology to make structures more efficient and safe by using less material while ensuring structural integrity. Additionally, this project aims to explore the efficacy of current optimization methods and potentially improve them through the implementation of machine learning methods.

| ![Kanarachos optimal Solution](Reading/Figures/Truss Solutions/Kanarachos_Opt.png) | ![TRUSS optimal solution](Reading/Figures/Truss Solutions/TRUSS_Opt.png) |
|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Kanarachos** Truss typology optimal Solution                                                            | **TRUSS** Truss typology optimal Solution                                                        |


## Basic tour of the Bayesian Optimization analysis

![Description of the GIF](Reading/Figures/Solution approach/TrussBOPT_EOP.gif)

### In detail,

## Results





## Repo structure and contents
- 口Reading includes much of the reference literature which was used as benchmark [Kanarachos](1-s2.0-S0045794916302036-main.pdf): the optimisation method which is being compared to a machine learning solution.
- **pyJive**: a finite element code that can compute the natural frequencies of a given truss design, which can be treated as black box model for this project
- **truss_bridge**: a directory with input files for the case of the project, including a [notebook](truss_bridge/truss_bridge.ipynb) with a demonstration of how to interact with the finite element code

The original paper and reference can be found here,
Investigation and reference results obtained with other optimization approaches, check this paper: [Kanarachos et al., 2017](https://dx.doi.org/10.1016/j.compstruc.2016.11.005)

### TRUSS notebook:
The Jupyter notebook contains the implementation of the Bayesian optimiser and the posterior analysis. It covers a from blank implementation, a solution through Botorch module and analysis on the covergence. 

### Presentation_final
A breakdown of the project and some of its detail can be found in [TrussBOPT.pptx](TRUSS1/TRUSS1/TrussBOPT_EOP.pptx).

### Useful links
The original publication:
- Approach undertaken by Kanarachos through a pure optimisation approach, check this paper for the publication: [https://dx.doi.org/10.1016/j.compstruc.2016.11.005](https://dx.doi.org/10.1016/j.compstruc.2016.11.005)

Tutorials on Bayesian optimization:
- [https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec](https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec)
- [https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization](https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization)
