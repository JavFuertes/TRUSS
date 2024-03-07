# Bayesian optimization for truss structures

![Multi-dimensional solution space]("https://www.mathworks.com/help/examples/stats/win64/ParellelBayeianOptimizationExample_01.png")

**Figure 1:** Visualisation of a Multi-dimensional solution space

Structures that make optimal use of the material they are made of reduces the cost and environmental impact of their construction as the amount of material required. Optimization of structural design is a challenging task because of the high number of design parameters and the relatively expensive evaluation of the suitability of any given design. Standard optimization techniques in high-dimensional design space require a very large number of possible designs that need to be evaluated. In structural analysis, where evaluating the objective function and checking the constraints involves the solution of a structural mechanics problem, e.g. with finite elements, this quickly becomes very expensive, even if the model is relatively simple from structural point of view. Bayesian optimization is a machine-learning-based optimization technique that aims to reduce the number of evaluations of the objective function through data-driven exploration of the design space with a probabilistic surrogate.

![TRUSS optimal solution](reading/Figures/solution_approach/TrussBOPT_formulation.png)

**Figure 2:** TRUSS optimisation problem formulation

## Objective & Description:

<span style="font-size: larger;"><B>Project Objective:</B></span> To find the optimal truss design

To achieve this project objective, we need to find the optimal set of nodal coordinates and cross-sectional properties. This will allow us to minimize the total weight of the structure while satisfying various constraints related to the structure's natural frequencies.This problem can be formulated in the following way,

$$
\min_{\theta,\gamma} \mathbb{E}_{\sim P_{\text{data}}}[L(\mathbf{\gamma };\theta)] \quad \text{(1)}
$$

$$
\text{OF} \left\{
\begin{array}{c|cccccc}
& \gamma_1 & \gamma_2 & \dots & \gamma_{14} \\ \hline
\theta_1 & f(\theta_1, \gamma_1) & f(\theta_1, \gamma_2) & \dots & f(\theta_1, \gamma_{14}) \\
\vdots  & \vdots  & \vdots  & \dots  & \vdots \\
\theta_5 & f(\theta_5, \gamma_1) & f(\theta_5, \gamma_2) & \dots & f(\theta_5, \gamma_{14})
\end{array} 
\right\} \quad \text{(2)}
$$

$$
\text{subject to:} \quad \left\{
\begin{array}{l}
\omega_1 \geq 20, \\
\omega_2 \geq 40, \\
\omega_3 \geq 60
\end{array}
\right. \quad \text{(3)}
$$

The obtained solution from the above formulation results in a truss of lower mass but structurally sound demonstrating the possibility to implement ML methods to improve the efficiency and safety of structures. Moreover, it also demonstrates the efficacy of expanding current optimization methods and through implementation of machine learning methods.

| ![Kanarachos optimal Solution](reading/Figures/truss_solutions/Kanarachos_Opt.png) | ![TRUSS optimal solution](reading/Figures/truss_solutions/TRUSS_Opt.png) |
|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
|**Figure 3:** **Kanarachos** truss typology optimal solution  | **Figure 4:** **TRUSS** truss typology optimal solution  |

## Breakdown of the Bayesian Optimization analysis

The solution method employed in the Bayesian optimizer involves several key steps and components. Initially, two classes are initialized: one for handling the bridge object and another for the optimizer. These classes manage specific methods pertinent to their respective processes, such as setting up paths and files for reading and writing, as well as initializing class attributes required for objective function calculations.

The optimizer class then proceeds to explore the solution space by randomly initiating and evaluating 19 solutions for a given number of iterations. During this phase, hyperparameters, especially those related to the Gaussian process such as the kernel exploration-exploitation tradeoff, are defined to guide the optimization process effectively.Two main classes are defined: the TRUSS class, which stores attributes and methods necessary for processing data with provided geometry and profile files, and the BOPT class, dedicated solely to optimizing the solution space. The BOPT class utilizes an exploration-exploitation function, which includes the acquisition function and random searching, to efficiently navigate the solution space.The objective function, a critical component of the optimization process, comprises two main parts. Firstly, it involves updating class attributes and computing mass and eigenfrequencies for new solution locations. Secondly, it computes the Loss Function, which includes frequency loss and mass loss terms. Frequency loss penalizes deviations from frequency constraints, while mass loss evaluates mass differences against predefined thresholds.Within the Bayesian Optimization Function, Gaussian processes play a central role in learning and modeling the objective function. Sampling from the Gaussian process aids in understanding uncertainty in exploration. The selection of the kernel hyperparameter is crucial for effective learning and exploration, and gradient descent methods are employed in conjunction with the Gaussian process to validate regions in the solution space.

Additionally, various considerations are taken into account in the optimization process. These include the optimizer architecture, type of Gaussian process and its likelihood, acquisition function, fitting strategies, and other factors crucial for optimizing the solution space effectively.Overall, the Bayesian optimizer utilizes a combination of random search, Gaussian processes, and gradient descent methods to actively learn and explore the solution space, with careful consideration of hyperparameters and optimization strategies to ensure efficient and effective optimization of structural engineering problems.

![Description of the GIF](reading/Figures/solution_approach/TrussBOPT_EOP.gif)

**Figure 5:** A description of the solution method through the optimiser and its different components. 

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
