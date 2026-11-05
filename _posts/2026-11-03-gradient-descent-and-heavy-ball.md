---
layout: post
title: "On Gradient Based Optimization - Part 1: Gradient Descent and Heavy Ball"
date: 2026-11-03
categories: [Optimization]
tags: [Gradient Descent, Numerical Optimization, Convex Function, Hessian Matrix]
excerpt: "An exploration of the mathematical foundations and computational behavior of gradient descent and heavy ball."
references: []
---

## Introduction

Consider the problem of minimizing a smooth *unconstrained* objective function \\( f( \mathbf{x} ) \\) with respect to a set of decision variables \\( \mathbf{x} \in \mathbb{R}^n \\):

\begin{equation}
  \min_{\mathbf{x}} f( \mathbf{x} )
\label{eq:unconstrained_obj_function}
\end{equation}

The solution of optimization problem \\(\eqref{eq:unconstrained_obj_function}\\) is given by the *optimal* point \\( \mathbf{x}^* \\), which is characterized by the property \\( \nabla f( \mathbf{x}^* ) =0 \\), i.e. the rate of change (evaluated at \\( \mathbf{x}^* \\) ) in any direction is zero.

## Gradient Descent

Gradient descent is an iterative optimization algorithm. Its update rule is given by \\(\mathbf{x}_{k+1} = \mathbf{x}_k - \gamma \nabla f(\mathbf{x}_k)\\). Rewritten as follow:

\\[
\mathbf{x}_{k+1}
=
\mathbf{x}_k-\gamma\nabla f(\mathbf{x}_k),
\qquad k=0,1,2,\ldots
\\]

### Stability of Gradient Descent

### Optimal Step Size

### Convergence Analysis

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

## Conclusion
