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

## Heavy Ball

### The Momentum Method

### Numerical Experiment

## Conclusion
