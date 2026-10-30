---
layout: post
title: "MathJax Rendering Test"
date: 2026-10-30
categories: [Optimization]
tags: [Mathematics, MathJax, Testing]
excerpt: "A temporary test article to verify mathematical rendering, equation numbering, alignment, and cross-referencing."
---

This is a temporary article for testing mathematical typesetting in Chimera Lab.

## 1. Inline mathematics

We begin with a quadratic objective function. Its gradient is \\( \nabla f(\mathbf{x}) \\), and its Hessian is \\( \nabla^2 f(\mathbf{x}) \\).

The gradient descent update is \\( \mathbf{x}_{k+1} = \mathbf{x}_k - \gamma \nabla f(\mathbf{x}_k) \\).

## 2. Display mathematics

The following expression should appear as a centered, unnumbered equation:

\\[
f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T Q\mathbf{x} - \mathbf{b}^T\mathbf{x} + c
\\]

## 3. Automatic equation numbering

The following equation should receive an automatically generated number.

\begin{equation}
f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T Q\mathbf{x} - \mathbf{b}^T\mathbf{x} + c
\label{eq:quadratic}
\end{equation}

Notice that we did not manually specify the equation number.

## 4. Aligned equations

Now let's test the `align` environment using the gradient of the quadratic objective function.

\begin{align}
\nabla f(\mathbf{x})
&= \frac{1}{2}(Q+Q^T)\mathbf{x} - \mathbf{b} \notag \\\\
&= Q\mathbf{x} - \mathbf{b}.
\label{eq:gradient}
\end{align}

where the second equality follows from the assumption that \(Q\) is symmetric.

## 5. Equation cross-referencing

We can refer to the quadratic objective function using Equation \\(\eqref{eq:quadratic}\\).

Similarly, Equation \\(\eqref{eq:gradient}\\) gives its gradient when \(Q\) is symmetric.

This concludes the MathJax rendering test.
