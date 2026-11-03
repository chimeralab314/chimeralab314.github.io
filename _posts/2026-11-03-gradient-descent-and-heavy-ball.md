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

Consider the problem of minimizing a smooth *unconstrained* objective function \\( f(x) \\) with respect to a set of decision variables \\( \mathbf{x} \in \mathbb{R}^n \\):

\\[
  \min_x f(x)
\\]

## Gradient Descent

Gradient descent is an iterative optimization algorithm. Its update rule is given by \\(\mathbf{x}_{k+1} = \mathbf{x}_k - \gamma \nabla f(\mathbf{x}_k)\\). Rewritten as follow:

\\[
\mathbf{x}_{k+1}
=
\mathbf{x}_k-\gamma\nabla f(\mathbf{x}_k)
\\]

### Stability of Gradient Descent

### Optimal Step Size

### Convergence Analysis

### Numerical Experiment

## Heavy Ball

### The Momentum Method

### Numerical Experiment

## Conclusion
