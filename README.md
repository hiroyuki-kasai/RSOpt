# RSOpt (Riemannian stochastic optimization algorithms)

Authors: [Hiroyuki Kasai](http://www.kasailab.com/kasai_e.htm), [Bamdev Mishra](https://bamdevmishra.in/), [Hiroyuki Sato](https://sites.google.com/site/hiroyukisatoeng/), [Pratik Jawanpuria](https://pratikjawanpuria.com/publications/)

Last page update: May 31, 2019

Latest version: 1.0.3 (see Release notes for more info) 

<br />

Intdocution
----------

Let f: M -> R be a smooth real-valued function on a **[Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold) M**. The target problem concerns a given model variable w on M, and is expressed as
min_{w in M} f(w) := 1/n sum_{i=1}^n f_i(w), where n is the total number of the elements. 

This problem has many applications; for example, in [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) and the subspace tracking problem, 
which is the set of r-dimensional linear subspaces in R^d. 
The low-rank [matrix comletion](https://en.wikipedia.org/wiki/Matrix_completion) problem and tensor completion problem are promising applications concerning the manifold of fixed-rank matrices/tensors. 
The [linear regression](https://en.wikipedia.org/wiki/Linear_regression) problem is also defined on the manifold of fixed-rank matrices. 

A popular choice of algorithms for solving this probem is the Riemannian gradient descent method, which calculates the Riemannian full gradient estimation for every iteration.
However, this estimation is computationally costly when n is extremely large. A popular alternative is the **Riemannian stochastic gradient descent algorithm (R-SGD)**, 
which extends the **[stochastic gradient descent algorithm](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD)** in the Euclidean space to the Riemannian manifold. 
As R-SGD calculates only one gradient for the i-th sample, the complexity per iteration is independent of the sample size n. 
Although R-SGD requires retraction and vector transport operations in every iteration, those calculation costs can be ignored when they are lower than those of the stochastic gradient; 
this applies to many important Riemannian optimization problems, including the low-rank tensor completion problem and the Riemannian centroid problem. 

Similar to SGD, R-SGD is hindered by a slow convergence rate due to a **decaying step size** sequence. To accelerate the rate of R-SGD, 
the **Riemannian stochastic variance reduced gradient algorithm (R-SVRG)** has been proposed; 
this technique reduces the variance of the stochastic gradient exploiting the finite-sum based on recent progress in **variance reduction** methods in the Euclidean space . 
One distinguished feature is reduction of the variance of noisy stochastic gradients by periodical full gradient estimations, which yields a linear convergence rate.
**Riemannian stochastic quasi-Newton algorithm with variance reduction algorithm (R-SQN-VR)** has also recently been proposed, where a stochastic quasi-Newton algorithm and the variance reduced methods are mutually combined. 
Furthermore, the **Riemannian stochastic recursive gradient algorithm (R-SRG)** has recently been also proposed to accelerate the convergence rate of R-SGD.

This RSOpt package provides the MATLAB implementation codes dedicated to those **stochastic** algorithms above. 

Note that various manifold algrithms on various manifolds are implemented in MATLAB toolbox [manopt](https://manopt.org/). The RSOpt codes are compliant to manopt. 
Also, please see [here](https://press.princeton.edu/titles/8586.html) for more comprehensive explanation of **optimization algorithms on matrix manifolds**.

<br />

Algorithms
----------

- **R-SGD (Riemannian stochastic gradient descent)** algorithm
    - S.Bonnabel, "[Stochastic gradient descent on Riemannian manifolds](https://ieeexplore.ieee.org/document/6487381/)," IEEE Trans. on Auto. Cont., 2013.
    
- **R-SVRG (Riemannian stochastic variance reduced gradient)** algorithm 
    - H.Sato, H.Kasai and B.Mishra, "Riemannian stochastic variance reduced gradient with retration and vector transport," [SIOPT2019](https://epubs.siam.org/doi/10.1137/17M1116787), [arXiv2017](https://arxiv.org/abs/1702.05594).
    - H.Kasai, H.Sato and B.Mishra, "[Riemannian stochastic variance reduced gradient on Grassmann manifold](http://opt-ml.org/papers/OPT2016_paper_13.pdf)," NIPS workshop OPT2016, 2016.
    - H.Zhang, S.J.Reddi and S.Sra, "[Fast stochastic optimization on Riemannian manifolds](http://papers.nips.cc/paper/6515-riemannian-svrg-fast-stochastic-optimization-on-riemannian-manifolds)," NIPS2016, 2016.

- **R-SRG (Riemannian stochastic recursive gradient)** algorithm
  - H.Kasai, H.Sato and B.Mishra, "[Riemannian stochastic recursive gradient algorithm](http://proceedings.mlr.press/v80/kasai18a.html)," ICML2018, 2018.

- **R-SQN-VR (Riemannian stochastic quasi-Newton algorithm with variance reduction)** (Not yet included) 
  - H.Kasai, H.Sato and B.Mishra, "[Riemannian stochastic quasi-Newton algorithm with variance reduction and its convergence analysis](http://proceedings.mlr.press/v84/kasai18a.html)," AISTATS2018, 2018.

- **RASA** (**R**iemannian **A**daptive **S**tochastic gradient **a**lgorithm on matrix manifolds) (Coming soon!) 
  - H.Kasai, P.Jawanpuria and B.Mishra, "Riemannian adaptive stochastic gradient algorithms on matrix manifolds," [ICML2019](http://proceedings.mlr.press/v97/kasai19a.html), 2019.


  
<br />


Folders and files
---------
<pre>
./                      - Top directory.
./README.md             - This readme file.
./run_me_first.m        - The scipt that you need to run first.
./demo.m                - Demonstration script to check and understand this package easily. 
|solvers/               - Contains various Riemannian stochastic optimization algorithms.
|tool/                  - Some auxiliary tools for this project.
|manopt/                - Contains manopt toolbox.
</pre>  
<br />
 

First to do
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```
<br />

Demonstration script
----------------------------
Run `demo` for computing the Riemannian centroid of **N** symmetric positive-definite (SPD) matrices of size **dxd**. This problem frequently appears in computer vision
problems such as visual object categorization and pose categorization. This demonstation handles N=500 and d=3.
```Matlab
demo; 
```

<br />
<img src="http://www.kasailab.com/public/github/RSOpt/images/RCProblem_N500_d3_integrate.png" width="900">
<br />

More plots
----------------------------
Run `show_centroid_plots` for the same Riemannian centroid problem. This scripts compares R-SGD, R-SVRG, R-SRG and R-SRG+ as well as batch algorithms including R-SD and R-CG. This scripts handles N=5000 and d=10.
```Matlab
show_centroid_plots; 
```
<br />
<img src="http://www.kasailab.com/public/github/RSOpt/images/RCProblem_N5000_d10_integrate.png" width="900">



<br />

License
---------------------
- The code is free and open source.
- The code should only be used for academic/research purposes.

<br />


Notes
---------------------
- The code is compliant to MATLAB toolbox [manopt](https://manopt.org/).


<br />

Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://www.kasailab.com/kasai_e.htm) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

<br />

Release Notes
--------------
* Version 1.0.3 (May 31, 2019)
    - Some paper informaiton are updated. 
* Version 1.0.2 (Sep. 13, 2018)
    - MC problem (with Jester dataset) example is added. 
* Version 1.0.1 (July 20, 2018)
    - Initial codes are available.
* Version 1.0.0 (July 12, 2018)
    - Initial version.  

