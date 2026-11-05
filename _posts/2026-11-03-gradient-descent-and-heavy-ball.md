---
layout: post
title: "On Gradient Based Optimization - Part 1: Gradient Descent and Heavy Ball"
date: 2026-11-03
categories: [Optimization]
tags: [Gradient Descent, Numerical Optimization, Convex Function, Hessian Matrix]
excerpt: "An exploration of the mathematical foundations and computational behavior of gradient descent and heavy ball."
references:
  - boyd2004convex
  - brunton2026optimization
  - ruder2016gradient
---

## Introduction

Consider the problem of minimizing a smooth *unconstrained* objective function \\( f( \mathbf{x} ) \\) with respect to a set of decision variables \\( \mathbf{x} \in \mathbb{R}^n \\):

\begin{equation}
  \min_{\mathbf{x}} f( \mathbf{x} )
\label{eq:unconstrained_obj_function}
\end{equation}

The solution of optimization problem \\(\eqref{eq:unconstrained_obj_function}\\) is given by the *optimal* point \\( \mathbf{x}^* \\), which is characterized by the property \\( \nabla f( \mathbf{x}^* ) =0 \\), i.e. the rate of change (evaluated at \\( \mathbf{x}^* \\) ) in any direction is zero. lorem ipsum dolor sit amet consectetur adipiscing elit lorem non fugiat nulla consequatur nihil quis exercitation cillum labore deserunt quidem laboris aut labore quibusdam quo et optio assumenda do expedita eos dolorem id eligendi nostrud consequatur nulla dolor ut quibusdam minus cupidatat dolor provident cupiditate enim est vel possimus omnis. lorem ipsum dolor sit amet consectetur adipiscing elit lorem non fugiat nulla consequatur nihil quis exercitation cillum labore deserunt quidem laboris aut labore quibusdam quo et optio assumenda do expedita eos dolorem id eligendi nostrud consequatur nulla dolor ut quibusdam minus cupidatat dolor provident cupiditate enim est vel possimus omnis.

## Gradient Descent

Gradient descent is an iterative optimization algorithm. Its update rule is given by \\(\mathbf{x}_{k+1} = \mathbf{x}_k - \gamma \nabla f(\mathbf{x}_k)\\). Rewritten as follow:

\\[
\mathbf{x}_{k+1}
=
\mathbf{x}_k-\gamma\nabla f(\mathbf{x}_k),
\qquad k=0,1,2,\ldots
\\]

Gradient-based optimization methods are widely used in convex optimization {% include citation.html key="boyd2004convex" %}. For some practical recommendation related to gradient descent, refer to {% include citation.html key="ruder2016gradient" %}. lorem ipsum dolor sit amet consectetur adipiscing elit lorem non fugiat nulla consequatur nihil quis exercitation cillum labore deserunt quidem laboris aut labore quibusdam quo et optio assumenda do expedita eos dolorem id eligendi nostrud consequatur nulla dolor ut quibusdam minus cupidatat dolor provident cupiditate enim est vel possimus omnis. lorem ipsum dolor sit amet consectetur adipiscing elit lorem non fugiat nulla consequatur nihil quis exercitation cillum labore deserunt quidem laboris aut labore quibusdam quo et optio assumenda do expedita eos dolorem id eligendi nostrud consequatur nulla dolor ut quibusdam minus cupidatat dolor provident cupiditate enim est vel possimus omnis.

### Stability of Gradient Descent

Given a convex quadratic function, we will examine how step size affect the stability of gradient descent {% include citation.html key="brunton2026optimization" %}.

For a detailed discussion, see [Stability of Gradient Descent](#stability-of-gradient-descent).

For a detailed discussion, see [Optimal Step Size](#optimal-step-size).

### Optimal Step Size

The optimal step size for gradient descent for a quadratic objective function is given by:

\begin{equation}
  \gamma^* = \frac{2}{ \lambda_{min} + \lambda_{max} }
\label{eq:optimal_step_size}
\end{equation}

### Convergence Analysis

As discussed in [Stability of Gradient Descent](#stability-of-gradient-descent), the choice of step size determines the convergence behavior. The optimal step size is given by equation \eqref{eq:optimal_step_size}, while the numerical behavior is illustrated in <a href="#figure-gd-contour"></a>.

### Numerical Experiment

#### Experiment 1

{% include figure.html
   id="gd-exp1-stepsize"
   src="/assets/images/articles/step-size-effect.png"
   alt="Plot showing the objective function's value vs iteration for different values of step sizes"
   caption="The objective function's value vs iteration with different step sizes."
%}

<a href="#figure-gd-exp1-stepsize"></a> illustrates the effect of different step sizes on the objective function's value vs iterations.

#### Experiment 2

{% include figure.html
   id="gd-contour"
   src="/assets/images/articles/contour-plot.png"
   alt="Contour plot showing gradient descent trajectories for two different step sizes"
   caption="Gradient descent trajectories with different step sizes. The red trajectory uses the optimal step size, while the blue trajectory uses a smaller step size."
%}

<a href="#figure-gd-contour"></a> illustrates the effect of different step sizes on the gradient descent trajectory.

{% include figure.html
   id="gd-objfunction-vs-iteration"
   src="/assets/images/articles/gd-second-plot.png"
   alt="Plot showing the objective function's value vs iteration for two different step sizes"
   caption="The objective function's value vs iteration with different step sizes. The red trajectory uses the optimal step size, while the blue trajectory uses a smaller step size."
%}

<a href="#figure-gd-objfunction-vs-iteration"></a> illustrates the effect of different step sizes on the objective function's value vs iterations.

#### Experiment 3

## Heavy Ball

### The Momentum Method

### Numerical Experiment

{% include reproducibility.html
   description="The numerical experiments and figures presented in this article were generated using Python. The complete implementation and experimental setup are available in the accompanying Jupyter notebook."
   url="https://github.com/wap314/Gradient-Descent-Experiment/blob/main/GD_experiment.ipynb"
%}

## Conclusion
