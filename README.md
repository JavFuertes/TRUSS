# Bayesian optimization for truss structures

Structures that make optimal use of the material they are made of reduces the cost and environmental impact of their construction as the amount of material required. Optimization of structural design is a challenging task because of the high number of design parameters and the relatively expensive evaluation of the suitability of any given design. Standard optimization techniques in high-dimensional design space require a very large number of possible designs that need to be evaluated. In structural analysis, where evaluating the objective function and checking the constraints involves the solution of a structural mechanics problem, e.g. with finite elements, this quickly becomes very expensive, even if the model is relatively simple from structural point of view. Bayesian optimization is a machine-learning-based optimization technique that aims to reduce the number of evaluations of the objective function through data-driven exploration of the design space with a probabilistic surrogate.

<h2 id="Background">
    <B>Objective & Description:</B><a class="anchor-link" href="#Background">&#182;</a>
</h2>

<div style="width: 60%; border-top: 4px solid #00B8C8; border-left: 4px solid #00B8C8; background-color: #FFFFFF; padding: 1em 1em 1em 1em; color: #24292E; margin: 10px 0 20px 0; box-sizing: border-box;">
    <div style="background-color: #00B8C8; color: white; padding: 0.2em 1em; margin: -1em -1em 0em -1em; font-size: 1.2em;"><strong>Project Objective:</strong> To find the optimal truss design</div>
    <p>
    To achieve so this project requires to find the optimal set of nodal coordinates and cross-sectional properties. Achieving so will allow to minimize as much as possible the total weight of the structure, while satisfying a number of constraints relating to the structures natural frequencies. Achieving a low mass solutions that also satisfies the natural frequencies constraints established demonstrates a methodology to make structures more efficient and safe since we are achieving to use less material in a way that still ensures the structural integrity of the structure. Moreover, this project will also aim to explore the efficacy of current optimisation methods and potentially improvements to be achieved from implementing machine learning methods.</p>
</div>

|              |              |
|--------------|--------------|
| ![Kanarachos optimal Solution](..\TRUSS1\TRUSS1\口Reading\Figures\Truss Solutions\Kanarachos_Opt.png) | ![TRUSS optimal solution](..\TRUSS1\TRUSS1\口Reading\Figures\Truss Solutions\TRUSS_Opt.png) |
| Kanarachos Truss typology optimal Solution | TRUSS Truss typology optimal Solution |

# Project Overview
This repository houses the artifacts and documentation related to our project, which focuses on Bayesian Optimization. The key components of this repository include a Jupyter notebook containing the Bayesian Optimization implementation, project planning document, and the final presentation materials.

## Basic tour of the Bayesian Optimization analysis

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
A small breakdown of the progress made in the project [TrussBOPT.pptx].

### Useful links
Tutorials on Bayesian optimization:
- [https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec](https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec)
- [https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization](https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization)
