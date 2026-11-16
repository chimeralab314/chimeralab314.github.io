---
layout: post
title: "On Gradient Based Optimization - Part 1: Gradient Descent and Heavy Ball"
date: 2026-11-03
categories: [Optimization]
tags: [Gradient Descent, Numerical Optimization, Convex Function, Hessian Matrix]
excerpt: "An exploration of the mathematical foundations and numerical behavior of gradient descent and heavy ball."
references:
  - boyd2004convex
  - brunton2026optimization
  - strang2019linearalgebra
  - ruder2016gradient
---

## 1. Introduction

Consider the problem of minimizing a smooth *unconstrained* objective function \\( f( \mathbf{x} ) \\) with respect to a set of decision variables \\( \mathbf{x} \in \mathbb{R}^n \\):

\begin{equation}
  \min_{\mathbf{x}} f( \mathbf{x} )
\label{eq:unconstrained_obj_function}
\end{equation}

The solution of optimization problem \\(\eqref{eq:unconstrained_obj_function}\\) is given by the *optimal* point \\( \mathbf{x}^* \\), which is characterized by zero gradient, \\( \nabla f( \mathbf{x}^* ) =0 \\), i.e. the rate of change in any direction is zero, and a positive semidefinite Hessian, \\( \mathbf{H}(\mathbf{x}^\*) = \nabla^2 f( \mathbf{x}^* ) \geq 0 \\), i.e. the curvature is nonnegative in any direction.

It is tempting to solve problem \\(\eqref{eq:unconstrained_obj_function}\\) by directly find point where \\( \nabla f( \mathbf{x} ) = 0 \\). However, lets consider a simple quadratic objective function:

\\[
  f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{A} \mathbf{x} + \mathbf{b}^T \mathbf{x} + c
\\]

with symmetric Hessian matrix \\( \mathbf{A} \\). The gradient is given by:

\\[
  \nabla f(\mathbf{x}) = \mathbf{A} \mathbf{x} + \mathbf{b}
\\]

By setting the gradient to zero, we find the optimal point \\( \mathbf{x}^* = - \mathbf{A}^{-1} \mathbf{b} \\). This looks deceptively simple, but when the dimension of \\( \mathbf{x} \in \mathbb{R}^n \\) is large (millions or billions), then \\( \mathbf{A}^{-1} \in \mathbb{R}^{n \times n} \\) is *intractable* to compute. In such cases, we prefer to use *iterative* algorithm to obtain sequence of points \\( \mathbf{x}_k \\) that eventually converge towards the *local* minimum \\( \mathbf{x}^* \\), instead of solve for \\( \mathbf{x}^* \\) directly. By using iterative algorithm, we can avoid the intractability of computing  \\( \mathbf{A}^{-1} \\).

What do we mean precisely by iterative algorithm ? We seek an update rule of the form:

\begin{equation}
  \mathbf{x}_{k+1} = \mathbf{F} ( \mathbf{x}_k )
\label{eq:update_rule}
\end{equation}

so that starting with initial guess \\( \mathbf{x}_0 \\), we can iterate through the function \\( \mathbf{F} \\) to iteratively decrease the objective function toward local minimum \\( \mathbf{x}^* \\):

\\[
   f(\mathbf{x}_0) \geq f(\mathbf{x}_1) \geq \ldots \geq f(\mathbf{x}_k) \geq \ldots \geq f(\mathbf{x^*})
\\]

At this point, we have the basic motivation to look for update rule in the form of equation \\(\eqref{eq:update_rule}\\). The next section will discuss the iterative algorithm in more detail.

## 2. Gradient Descent

How do we design the update rule \\(\eqref{eq:update_rule}\\) such that the sequence \\( \mathbf{x}_k \\) converge toward the local minimum \\( \mathbf{x}^* \\) ? We need to find a direction in the domain of the objective function such that the objective function is decreasing. Note that the domain of the objective function is \\( n \\)-dimensional. Thus, it is not trivial to choose a certain direction where the objective function decrease.

One simple idea is to take the direction of *steepest descent*, i.e. the direction of the greatest rate of decrease. To illuminate this concept, recall first that the *directional derivative* of a function \\( f( \mathbf{x} ) \\) in the direction of unit vector \\( u \\) is simply the component (scalar projection) of gradient vector \\( \nabla f( \mathbf{x} ) \\) onto unit vector \\( u \\):

\\[
   D_u f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot u
\\]

Thus, once we compute the gradient vector \\( \nabla f( \mathbf{x} ) \\), we can calculate directional derivative in any direction. Among all possible direction, the directional derivative reach its maximum value when unit vector \\( u \\) is in the direction of the gradient vector  \\( \nabla f( \mathbf{x} ) \\). This can be shown as follow:

\begin{aligned}
  D_u f(\mathbf{x}) 
  &= \nabla f(\mathbf{x}) \cdot u \cr
  &= \\| \nabla f(\mathbf{x}) \\| \\| u \\| \cos \theta \cr
  &= \\| \nabla f(\mathbf{x}) \\| \cos \theta
\end{aligned}

Thus, the directional derivative \\( D_u f(\mathbf{x}) \\) reach its maximum value \\( \\| \nabla f( \mathbf{x} \\| \\), when \\( \theta = 0 \\), i.e. unit vector \\( u \\) is in the direction of gradient vector \\( \nabla f(\mathbf{x}) \\). This simple result shows that the gradient vector \\( \nabla f( \mathbf{x} ) \\) points to the direction of *steepest ascent*, i.e. the direction of the greatest rate of increase. Hence, if we take the negative of gradient vector, \\( -\nabla f( \mathbf{x} ) \\), it points to the direction of steepest descent ! If we follow the direction of steepest descent, then starting from initial guess \\( \mathbf{x}_0 \\), we can obtain sequence of points \\( \mathbf{x}_k \\) such that the objective function is decreasing. In short: gradient descent seeks local minimum of an objective function \\( f( \mathbf{x} ) \\) by taking iterative steps in the direction of negative gradient, \\( -\nabla f( \mathbf{x} ) \\).

Physically, we can roughly interpret the negative of gradient vector, \\( -\nabla f( \mathbf{x} ) \\), as *downhill force* that pulls \\( \mathbf{x}_k \\) toward local minimum. Informally, the update rule for gradient descent can be written as follow:

\\[
  \text{New position} = \text{Old position} + (\text{Step size}) (\text{Downhill force})
\\]

Mathematically, gradient descent is given by:

\begin{equation}
  \mathbf{x}_{k+1} = \mathbf{x}_k-\gamma\nabla f(\mathbf{x}_k)
\label{eq:gd_update_rule}
\end{equation}

Update rule \\(\eqref{eq:gd_update_rule}\\) is the iterative algorithm we've been looking for.

What is the role of step size \\( \gamma \\) ? Note that the direction of negative gradient is *local* information, since gradient depends locally on where you are. This simply means that given a point \\( \mathbf{x}_k \\), the local direction of negative gradient, \\( -\nabla f( \mathbf{x}_k ) \\), does not necessarily point toward the local minimum \\( \mathbf{x}^* \\). Hence, the gradient descent algorithm must carefully take a step to achieve the local minimum. This is where the role of step size becomes important.

The step size \\( \gamma \\) determine the magnitude of the step taken by gradient descent. If step size is too large, then the iteration may overshoot the minimum, causing oscillation, or worse, diverge. On the other hand, if step size is too small, the iteration will converge too slowly. In other words, the step size determine *stability* and *convergence behaviour* of gradient descent algorithm.

The algorithm of gradient descent is described as follow:

<table style="width:100%; border-collapse:collapse; border-top:2px solid black; border-bottom:2px solid black; font-size:1.05em;">

  <thead>
    <tr style="border-bottom:2px solid black;">
      <th colspan="2" style="text-align:left; padding:4px 8px; font-weight:normal;">
        <strong>Algorithm 1</strong> &nbsp; Gradient Descent Method
      </th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td style="width:45px; padding:3px 10px; vertical-align:top;">1:</td>
      <td style="padding:3px 8px;"><strong>Input</strong>: Initial iterate \(\mathbf{x}_0\), maximum iteration \( N \) and step size \( \gamma \).</td>
    </tr>

    <tr>
      <td style="padding:3px 10px; vertical-align:top;">2:</td>
      <td style="padding:3px 8px;"><strong>for</strong> \(k=1,2,\ldots\,N\) <strong>do</strong></td>
    </tr>

    <tr>
      <td style="padding:3px 10px; vertical-align:top;">4:</td>
      <td style="padding:3px 8px 3px 35px;">
        Compute gradient of the objective function \(g_k = \nabla f(\mathbf{x}_k) \).
      </td>
    </tr>

    <tr>
      <td style="padding:3px 10px; vertical-align:top;">6:</td>
      <td style="padding:3px 8px 3px 35px;">
        Set the new iterate as
        \(\mathbf{x}_{k+1}\leftarrow \mathbf{x}_k-\gamma g_k\).
      </td>
    </tr>

    <tr>
      <td style="padding:3px 10px; vertical-align:top;">7:</td>
      <td style="padding:3px 8px;"><strong>end for</strong></td>
    </tr>
  </tbody>
</table>

Gradient-based optimization methods are widely used in convex optimization {% include citation.html key="boyd2004convex" %}{% include citation.html key="brunton2026optimization" %}. For some practical recommendation related to gradient descent, refer to Ruder {% include citation.html key="ruder2016gradient" %}.

### 2.1. Stability of Gradient Descent

In this subsection, we will explore the stability of gradient descent on a simple quadratic objective function:

\\[
  f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{A} \mathbf{x} + \mathbf{b}^T \mathbf{x} + c
\\]

with symmetric Hessian matrix \\( \mathbf{A} \\). The gradient is given by:

\\[
  \nabla f(\mathbf{x}) = \mathbf{A} \mathbf{x} + \mathbf{b}
\\]

For quadratic objective function, the gradient descent update rule:

\\[
  \mathbf{x}_{k+1} = \mathbf{x}_k-\gamma \nabla f(\mathbf{x}_k)
\\]

becomes

\begin{equation}
\begin{aligned}
  \mathbf{x}_{k+1}
  &= \mathbf{x}_k - \gamma ( \mathbf{A} \mathbf{x}_k + \mathbf{b} ) \cr
  &= ( \mathbf{I} - \gamma \mathbf{A} ) \mathbf{x}_k - \gamma \mathbf{b}
\label{eq:gd_on_quadratic}
\end{aligned}
\end{equation}

Equation \\( \eqref{eq:gd_on_quadratic} \\) implies that gradient descent algorithm can be seen as *dynamical system* with state variable \\( \mathbf{x}_k \\), where the stability properties of gradient descent depend on the step size \\( \gamma \\) and the eigenvalues of \\( \mathbf{A} \\). 

Note that the eigenvalues \\( \lambda_i \\) of the Hessian matrix \\( \mathbf{A} \\) represent the curvature of the objective function \\( f \\) in the eigenvector direction. The minimum curvature is given by \\( \lambda_{\min} \\), while the maximum curvature is given by \\( \lambda_{\max} \\).

The gradient descent algorithm is stable if *all eigenvalues* of \\( \mathbf{I} - \gamma \mathbf{A} \\) are stable. For discrete-time dynamical system \\( \eqref{eq:gd_on_quadratic} \\), stability means that all eigenvalues of \\( \mathbf{I} - \gamma \mathbf{A} \\) have *magnitude less than one*, which guarantee stability of the iteration: \\( \mathbf{x}_k \rightarrow \mathbf{x}^* = -\mathbf{A}^{-1} \mathbf{b} \\).

If matrix \\( \mathbf{A} \\) is diagonalizable, then there exist an invertible matrix of eigenvectors \\( \mathbf{T} \\), such that \\( \mathbf{T}^{-1} \mathbf{A} \mathbf{T} = \mathbf{D}  \\), where \\( \mathbf{D} = \text{diag}( \lambda_1, \ldots, \lambda_n ) \\). In this case, we have that:

\begin{aligned}
  \mathbf{T}^{-1} ( \mathbf{I} - \gamma \mathbf{A} ) \mathbf{T}
  &= \mathbf{I} - \gamma ( \mathbf{T}^{-1} \mathbf{A} \mathbf{T} ) \cr
  &= \mathbf{I} - \gamma \mathbf{D} \cr
  &= \text{diag} ( 1-\gamma \lambda_1, \ldots, 1-\gamma \lambda_n )
\end{aligned}

Thus, to guarantee stability of gradient descent update rule \\( \eqref{eq:gd_on_quadratic} \\), it is required that:

\\[
  \lvert 1-\gamma \lambda_i \rvert < 1, \forall i
\\]

The stability condition above can be rewritten as follow:

\\[
  -1 < 1-\gamma \lambda_i < 1 \quad \Rightarrow  \quad -2 < -\gamma \lambda_i < 0 \quad \Rightarrow \quad 0 < \gamma < \frac{2}{\lambda_i}
\\]

Note that the inequality above must be satisfied for all eigenvalues \\( \lambda_i \\). Since \\( \lambda_i \leq \lambda_{\max} \\), for all \\( i \\), we have that:

\begin{equation}
  \quad 0 < \gamma < \frac{2}{\lambda_{\max}}
\label{eq:stable_step_size}
\end{equation}

The inequality \\( \eqref{eq:stable_step_size} \\) provide the tightest interval that contains stable step sizes \\( \gamma \\). Thus, we have determined the range of stable step sizes \\( \gamma \\). Note that the interval of stable size is an open interval, where the infimum is zero and the supremum is given by \\( 2/\lambda_{\max} \\). Thus, inequality \\( \eqref{eq:stable_step_size} \\) shows that maximum curvature of the objective function determine the range of stable step sizes. When the maximum curvature is large, the range of stable step size is tight. When the maximum curvature is small, the range of stable step size is large.

Intuitively, formula \\( \eqref{eq:stable_step_size} \\) makes very much sense. When the curvature is large, overly large step size will lead to barely stable or unstable behaviour where the gradient step overshoot the minimum and oscillate. Thus, the range of step size should be narrow to avoid instability. On the other hand, when the curvature is small, then it is safe to take large step toward the minimum. Hence, the range of stepsize can be wide for faster convergence.

### 2.2. Optimal Step Size

The inequality \\( \eqref{eq:stable_step_size} \\) has provide us with the range of stable step sizes \\( \gamma \\). As long as our choice of \\( \gamma \\) lies within the range given by \\( \eqref{eq:stable_step_size} \\), then the gradient descent algorithm is guaranteed to be stable. One questien arise: among those value of \\( \gamma \\) in the interval \\( \eqref{eq:stable_step_size} \\), what is the value of \\( \gamma \\) that gives *optimal* behaviour ? Now we turn to the problem of determining the optimal step size \\( \gamma^* \\) for *fastest convergence* of gradient descent algorithm.

Recall the update rule \\( \eqref{eq:gd_on_quadratic} \\):

\\[
  \mathbf{x}_{k+1} = ( \mathbf{I} - \gamma \mathbf{A} ) \mathbf{x}_k - \gamma \mathbf{b}
\\]

Let's define the *error* relative to the local minimum as \\( \mathbf{e}_k = \mathbf{x}_k-\mathbf{x}^{*} \\). By definition:

\\[
  \mathbf{e}\_{k+1} = \mathbf{x}_{k+1} - \mathbf{x}^{*}
\\]

Substitute update rule \\( \eqref{eq:gd_on_quadratic} \\) for \\( \mathbf{x}_{k+1} \\):

\\[
\mathbf{e}_{k+1}
=
(\mathbf{I}-\gamma\mathbf{A})\mathbf{x}_k
-\gamma\mathbf{b}
-\mathbf{x}^{*}
\\]

Since \\( \mathbf{e}_k = \mathbf{x}_k-\mathbf{x}^* \\), then we have that \\( \mathbf{x}_k =  \mathbf{e}_k + \mathbf{x}^{*}\\). Substituting for \\( \mathbf{x}_k \\), then expand and rearrange:

\begin{aligned}
  \mathbf{e}_{k+1}
  &= (\mathbf{I}-\gamma\mathbf{A})( \mathbf{e}_k + \mathbf{x}^* ) -\gamma\mathbf{b} -\mathbf{x}^* \cr
  &= (\mathbf{I}-\gamma\mathbf{A})\mathbf{e}_k + (\mathbf{I}-\gamma\mathbf{A} ) \mathbf{x}^* -\gamma \mathbf{b}-\mathbf{x}^* \cr
  &= (\mathbf{I}-\gamma\mathbf{A})\mathbf{e}_k - \gamma ( \mathbf{A} \mathbf{x}^* + \mathbf{b} )
\end{aligned}

Note that at the local minimum, the gradient must vanish:

\\[
  \nabla f(\mathbf{x}^\*) = \mathbf{A} \mathbf{x}^* + \mathbf{b} = 0 
\\]

Therefore we have that:

\\[
  \mathbf{e}_{k+1} = ( \mathbf{I} - \gamma \mathbf{A} ) \mathbf{e}_k \quad \Rightarrow  \quad \mathbf{e}_k = ( \mathbf{I} - \gamma \mathbf{A} )^k \mathbf{e}_0
\\]

Suppose \\( v_i \\) is an eigenvector of \\( \mathbf{A} \\), then it can be shown that:

\begin{aligned}
  ( \mathbf{I} - \gamma \mathbf{A} ) v_i
  &= v_i - \gamma \mathbf{A} v_i \cr
  &= v_i - \gamma \lambda_i v_i \cr
  &= ( 1-\gamma \lambda_i ) v_i
\end{aligned}

Thus, \\( v_i \\) is also an eigenvector of \\( \mathbf{I} - \gamma \mathbf{A} \\) with the corresponding eigenvalue \\( 1-\gamma \lambda_i \\).

Let the initial error vector be described as linear combination of the eigenvector: \\( \mathbf{e}_0 = \sum_i^n c_i v_i \\). Hence, the error at \\( k\\) can be described as follow:

\begin{aligned}
  \mathbf{e}_k 
  &= ( \mathbf{I} - \gamma \mathbf{A} )^k \mathbf{e}_0 \cr
  &= ( \mathbf{I} - \gamma \mathbf{A} )^k \sum_i^n c_i v_i \cr
  &= \sum_i^n c_i ( \mathbf{I} - \gamma \mathbf{A} )^k v_i \cr
  &= \sum_i^n c_i ( 1 - \gamma \lambda_i )^k v_i \cr
  &= c_1 ( 1 - \gamma \lambda_1 )^k v_1 + \ldots + c_n ( 1 - \gamma \lambda_n )^k v_n
\end{aligned}

The equation above implies that fast-decaying eigenvalue will go to zero first, leaving behind slow-decaying eigenvalue. Thus, *the convergence rate of gradient descent is determined by the slowest-decaying eigenvalue*, i.e. \\( \max_i  \| 1-\gamma \lambda_i \| \\), which is the *spectral radius* of \\( \mathbf{I} - \gamma \mathbf{A} \\):

\begin{equation}
  \rho ( \mathbf{I} - \gamma \mathbf{A} ) = \max_i  \| 1-\gamma \lambda_i \|
\label{eq:spectral_radius}
\end{equation}

Therefore, to minimize the convergence rate, we choose \\( \gamma \\) such that \\( \rho ( \mathbf{I} - \gamma \mathbf{A}) \\) is as small as possible. Thus the optimal step size is given by:

\\[
  \gamma^* = \arg\min_{\gamma} \rho ( \mathbf{I} - \gamma \mathbf{A} )
\\]

For simplicity, suppose the Hessian matrix is *positive definite*, then the eigenvalues of \\( \mathbf{A} \\) satisfies the condition:

\\[
  0 < \lambda_{\min} \leq \lambda_i \leq \ldots \leq \lambda_{\max}, \forall i
\\]

Recall that the eigenvalues of the iteration matrix are

\\[
  \mu_i = 1 - \gamma \lambda_i
\\]

Since \\( \gamma > 0 \\), increasing \\( \lambda_i \\) will decreases \\( \mu_i \\). Thus, the extreme eigenvalues become:

\\[
  \mu_{\min} = 1-\gamma \lambda_{\max} \quad \text{and} \quad \mu_{\max} = 1-\gamma \lambda_{\min}
\\]

The two extreme eigenvalues determine the spectral radius:

\begin{equation}
  \rho ( \mathbf{I} - \gamma \mathbf{A}) = \max ( \| 1-\gamma \lambda_{\max} \|, \| 1-\gamma \lambda_{\min} \| )
\label{eq:spectral_radius_minmax_eigenvalue}
\end{equation}

Remember that our objective is to minimize the spectral radius. Since spectral radius consider the largest absolut eigenvalue \\( \|\mu_i\| \\), we can choose the optimal step size by *place the two extreme eigenvalues at equal distances on opposite sides of zero*. This is equivalent to minimizing the maximum distance of the iteration matrix eigenvalues from zero. Mathematically, it can be written as follow:

\\[
   \| 1-\gamma \lambda_{\min} \| = \| 1-\gamma \lambda_{\max} \|
\\]

For the optimal step size, the two extreme  eigenvalues must have opposite signs:

\\[
  1-\gamma^* \lambda_{\min} = -(1-\gamma^* \lambda_{\max})
\\]

After rearranging, we have the formula for optimal step size:

\begin{equation}
  \gamma^* = \frac{2}{\lambda_{\min} + \lambda_{\max}}
\label{eq:eq_optimal_gamma}
\end{equation}

The optimal step size \\( \gamma^* \\) minimizes the spectral radius \\( \rho ( \mathbf{I}- \gamma \mathbf{A} ) \\), leading to the fastest convergence for the quadratic problem.

Now we need to calculate the spectral radius when \\( \gamma = \gamma^* \\). Recall that the two extreme eigenvalues determine the spectral radius:

\\[
  \rho ( \mathbf{I} - \gamma \mathbf{A}) = \max ( \| 1-\gamma \lambda_{\max} \|, \| 1-\gamma \lambda_{\min} \| )
\\]

Substitute the optimal step size \\( \gamma^* \\) from equation \\( \eqref{eq:eq_optimal_gamma} \\) and using the fact that \\( \lambda_{\max} \geq \lambda_{\min} > 0\\):

\begin{aligned}
  \rho ( \mathbf{I}- \gamma^* \mathbf{A} ) 
  &= \max \left( \left\lvert 1- \frac{2\lambda_{\min}}{\lambda_{\min} + \lambda_{\max}} \right\rvert, \left\lvert 1- \frac{2\lambda_{\max}}{\lambda_{\min} + \lambda_{\max}} \right\rvert \right) \cr
  &= \max \left( \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\min} + \lambda_{\max}}, \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\min} + \lambda_{\max}} \right)
\end{aligned}

Both terms inside the parenthesis are equal. Therefore, their maximum is simply:

\\[
  \rho ( \mathbf{I}- \gamma^* \mathbf{A} ) 
  = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\min} + \lambda_{\max}}
\\]

The equation above is the resulting minimum spectral radius when we use the optimal step size \\( \gamma^* \\).

The *condition number* of a symmetric positive definite matrix is given by:

\\[
  \kappa(\mathbf{A}) = \frac{\lambda_{\max}}{\lambda_{\min}}
\\]

We can divide our numerator and denumerator of the minimum spectral radius by \\( \lambda_{\min} \\) and express it in terms of condition number:

\begin{equation}
\begin{aligned}
  \rho ( \mathbf{I} - \gamma^* \mathbf{A}) 
  &= \frac{(\lambda_{\max}/\lambda_{\min}) - 1}{(\lambda_{\max}/\lambda_{\min}) + 1} \cr
  &= \frac{ \kappa(\mathbf{A}) - 1 }{\kappa(\mathbf{A}) + 1}
\label{eq:min_spectral_radius}
\end{aligned}
\end{equation}

Equation \\( \eqref{eq:min_spectral_radius} \\) gives us a very interesting insight: *the convergence rate of gradient descent with optimal step size is directly determined by the condition number of the Hessian matrix*. If \\( \kappa(\mathbf{A}) \\) is close to one, then the minimum spectral radius is close to zero, thus convergence is fast. If \\( (\mathbf{A}) \\) is very large, then the minimum spectral radius approaches one, thus convergence is slow.

### 2.3. Convergence Analysis

In this subsection, we describe in detail convergence analysis of gradient descent following the presentation by Boyd & Vandenberghe {% include citation.html key="boyd2004convex" %} and Strang {% include citation.html key="strang2019linearalgebra" %}.

We consider the case of *strongly convex* objective function \\( f( \mathbf{x} ) \\) in \\( n \\) dimensions. For strongly convex function, the eigenvalues of Hessian matrix \\( \mathbf{H}(\mathbf{x}) = \nabla^2 f(\mathbf{x}) \\) are between \\( 0 < m \leq \lambda_i \leq M \\), for all \\( \mathbf{x} \\).

Recall that the gradient update rule is \\( \mathbf{x}_{k+1} = \mathbf{x}_k - \gamma \nabla f(\mathbf{x}_k) \\). We approximate the objective function \\( f(\mathbf{x}) \\) by its second order Taylor series:

\begin{equation}
  f(\mathbf{x}\_{k+1}) \approx f(\mathbf{x}\_k) + \nabla f(\mathbf{x}\_k)^T (\mathbf{x}\_{k+1} - \mathbf{x}\_k) + \frac{1}{2} (\mathbf{x}\_{k+1} - \mathbf{x}\_k)^T \mathbf{H} (\mathbf{x}_{k+1} - \mathbf{x}_k)
\label{eq:strongly_convex_Taylor_approx}
\end{equation}

Let \\( d=\mathbf{x}\_{k+1} - \mathbf{x}\_k \\), then \\( (\mathbf{x}\_{k+1} - \mathbf{x}\_k)^T \mathbf{H} (\mathbf{x}_{k+1} - \mathbf{x}_k) = d^T \mathbf{H} d \\). Using the eigendecomposition of symmetric \\( \mathbf{H} \\):

\begin{aligned}
  d^T \mathbf{H} d
  &= d^T Q \Lambda Q^T d \cr
  &= (Q^T d)^T \Lambda  (Q^T d) \cr
  &= z^T \Lambda z
\end{aligned}

Since \\( \Lambda = \text{diag}( \lambda_1, \ldots, \lambda_n) \\) and \\( 0 < \lambda_i \leq M \\), then:

\begin{aligned}
  d^T \mathbf{H} d
  &= z^T \Lambda z \cr
  &= \lambda_1 z_1^2 + \lambda_2 z_2^2 + \ldots + \lambda_n z_n^2 \cr
  &\leq M \lVert z \rVert^2
\end{aligned}

Note that \\( z=Q^T d \Rightarrow \\| z \\|^2 = z^Tz = \\| d \\|^2 \\). Thus, the upper bound on the quadratic form of symmetric positive definite Hessian is given by:

\\[
  d^T \mathbf{H} d \leq M \\| d \\|^2
\\]

Substitute the upper bound on \\( \eqref{eq:strongly_convex_Taylor_approx} \\):

\begin{equation}
  f(\mathbf{x}\_{k+1}) \leq f(\mathbf{x}\_k) + \nabla f(\mathbf{x}\_k)^T (\mathbf{x}\_{k+1} - \mathbf{x}\_k) + \frac{M}{2} \\| \mathbf{x}_{k+1} - \mathbf{x}_k \\|^2
\label{eq:strongly_convex_upper_bound}
\end{equation}

Recall that the gradient descent update rule is given by \\( \mathbf{x}\_{k+1} = \mathbf{x}\_k - \gamma \nabla f(\mathbf{x}\_k) \\). Hence we have: \\( \mathbf{x}_{k+1} - \mathbf{x}\_k = - \gamma \nabla f(\mathbf{x}\_k) \\). Substitute this expression to \\( \eqref{eq:strongly_convex_upper_bound} \\):

\begin{equation}
\begin{aligned}
  f(\mathbf{x}_{k+1}) 
  &\leq f(\mathbf{x}_k) - \gamma \\|  \nabla f(\mathbf{x})\_k \\|^2 + \frac{M}{2} \gamma^2 \\|  \nabla f(\mathbf{x})\_k \\|^2 \cr
  &= f(\mathbf{x}_k) + \\|  \nabla f(\mathbf{x})\_k \\|^2 \left( -\gamma + \frac{M}{2} \gamma^2 \right)
\end{aligned}
\label{eq:strong_convex_upperbound_gamma}
\end{equation}

Now we need to look for step size \\( \gamma \\) that minimize the upper bound in \\( \eqref{eq:strong_convex_upperbound_gamma} \\). The right hand side of \\( \eqref{eq:strong_convex_upperbound_gamma} \\) is a function of \\( \gamma \\): \\( g(\gamma) = -\gamma + \frac{M}{2} \gamma^2 \\). The step size \\( \gamma \\) that minimize \\( g(\gamma) \\) can be calculated by setting the derivative of \\( g(\gamma) \\) with respect to \\( \gamma \\) equal to zero:

\\[
  M \gamma -1 =0 \Rightarrow \gamma = \frac{1}{M}
\\]

Basically we choose \\( \gamma \\) such that the step size is small when the curvature is large. Substitute our selection of \\( \gamma \\) to \\( \eqref{eq:strong_convex_upperbound_gamma} \\), then we have the upper bound as follow:

\begin{equation}
  f(\mathbf{x}\_{k+1}) \leq f(\mathbf{x}\_k) - \frac{1}{2M} \\| \nabla f(\mathbf{x}\_k) \\|^2
\label{eq:strong_convex_upperbound_M}
\end{equation}

The inequality \\( \eqref{eq:strong_convex_upperbound_M} \\) tells us that as we move from \\( \mathbf{x}\_k \\) to \\( \mathbf{x}_{k+1} \\) by following negative gradient direction with step size \\( \gamma = 1/M \\), the objective function *decrease* from \\( f(\mathbf{x}\_k) \\) to \\( f(\mathbf{x}\_{k+1}) \\) *at least* by amount \\( 1/2M \\| \nabla f(\mathbf{x}_k)  \\|^2  > 0\\). The inequality \\( \eqref{eq:strong_convex_upperbound_M} \\) becomes equality when \\( \mathbf{x}_k = \mathbf{x}^* \Rightarrow \nabla f(\mathbf{x}^*) = 0 \\).

A parallel argument  uses \\( m \\) insted of \\( M \\) leads to the following inequality:

\begin{equation}
  f(\mathbf{x}^*) \geq f(\mathbf{x}_k) - \frac{1}{2m} \\| \nabla f(\mathbf{x}_k)  \\|^2
\label{eq:strong_convex_lower_bound}
\end{equation}

which can be re-written as follow:

\\[
  f(\mathbf{x}_k) - f(\mathbf{x}^*) \leq \frac{1}{2m} \\| \nabla f(\mathbf{x}_k)  \\|^2
\\]

In words, inequality \\( \eqref{eq:strong_convex_lower_bound} \\) tells us that the error \\( f(\mathbf{x}_k) - f(\mathbf{x}^\*) \\) cannot be *arbitrarily small* when we're still far from the minimum.

Multiply \\( \eqref{eq:strong_convex_upperbound_M} \\) by \\( M \\) and \\( \eqref{eq:strong_convex_lower_bound} \\) by \\( m \\) and subtract to remove \\( \\| \nabla f(\mathbf{x}_k)  \\|^2 \\). The results can be rewritten as follow:

\\[
  f(\mathbf{x}\_{k+1}) - f(\mathbf{x}^\*) \leq \left( 1-\frac{m}{M} \right) \left( f(\mathbf{x}_k) - f(\mathbf{x}^\*) \right)
\\]

Let's define the objective function's error  \\( e\_k = f(\mathbf{x}\_{k}) - f(\mathbf{x}^\*) \\), the inequality above can rewritten as follow:

\begin{equation}
  e_{k+1} \leq \left( 1-\frac{m}{M} \right) e_k
\label{eq:error_linear_conv}
\end{equation}

Equation \\( \eqref{eq:error_linear_conv} \\) shows that every negative gradient step decrease the error \\( e_k \\) at least by amount \\( 1-\frac{m}{M} \\). This is *linear convergence* where every step decrease the objective function by *the same constant factor*. 

If we start with initial error \\( e_0 \\), inequality \\( \eqref{eq:error_linear_conv} \\) implies that:

\begin{equation}
  e\_k \leq \left( 1 - \frac{m}{M} \right)^k e_0 \quad \Rightarrow \quad e\_k = O(\rho^k) 
\label{eq:error_geometric_decay}
\end{equation}

The inequality \\( \eqref{eq:error_geometric_decay} \\) tells us that the error \\( e_k \\) decreases *geometrically* with the number of iterations \\( k \\), with contraction factor \\( \rho = 1 - \frac{m}{M} \\). Note that when \\( m/M \\) is small (or \\( \kappa(\mathbf{H}) = M/m \\) is large), then the convergence is slow. On the other hand, when \\( m/M \\) close to one (similarly \\(  \kappa(\mathbf{H}) = M/m \\) close to one), then the convergence is fast. Note that this result is in agreement with the result we have from minimum spectral radius \\( \eqref{eq:min_spectral_radius} \\), although we use *different formula for step size* \\( \gamma \\). Therefore we can conclude that *geometry* of the objective function, in the form of condition number of Hessian matrix \\( \kappa(\mathbf{H}) \\), determine convergence rate of gradient descent for a given choice of step size \\( \gamma \\).

### 2.4. Numerical Experiment

In this subsection, we will apply gradient descent algorithm to quadratic objective function in various dimension. We will observe and examine the algorithm's behavior such as stability, convergence rate and scalability.

#### Experiment 1
We will start our experiment by applying gradient descent algorithm to a simple one-dimensional quadratic objective function given by:

\begin{equation}
  f(x) = x^2 + 2x + 5
\label{eq:1d_quadratic_function}
\end{equation}

The minimum of function \\( \eqref{eq:1d_quadratic_function} \\) is \\( f(x^\*)=4\\), where \\(x^*=-1\\). We try several step sizes to see the effect on the algorithm's stability and convergence rate. We choose some step size values: \\( \gamma=1, \gamma=0.01 \\) and \\( \gamma=0.4 \\).

{% include figure.html
   id="gd-exp1-stepsize"
   src="/assets/images/articles/step-size-effect.png"
   alt="Plot showing the objective function's value vs iteration for different values of step sizes"
   caption="The objective function's value vs iteration with different step sizes."
%}

<a href="#figure-gd-exp1-stepsize"></a> shows the effect of different step sizes on the objective function's value vs iterations. The figure shows some interesting results. For step-size \\(\gamma=1\\), the iteration is nearly unstable. It is neither converge nor diverge, but stuck at the initial function value. This suggest that \\(\gamma=1\\) is too large. For step-size \\(\gamma=0.01\\), the iteration seems to converge to the minimum but it is way too slow. This indicates that \\(\gamma=0.01\\) is too small. The last value, \\(\gamma=0.4\\) show relatively fast convergence at the beginning and then reach the minimum after two iteration (approximately). Hence for this simple case, step-size \\(\gamma=0.4\\) is the optimal value among three step-size values the we experimented with.

This simple experiment confirm our mathematical analysis that step size determine the stability and convergence rate of the gradient descent algorithm. If step size is too large, the iteration will be barely stable or even unstable. On the other hand, if step size is too small, the iteration will converge too slowly. The optimal step size lies somewhere between the two and it lead to fast convergence.

#### Experiment 2

In this experiment, we will explore the effect of step size on the stability of gradient descent in a more sophisticated manner. Consider a two-dimensional quadratic objective function:

\begin{equation}
f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^{T} \mathbf{A} \mathbf{x},
\qquad \text{where} \qquad
\mathbf{A} = \begin{pmatrix}
2 & 1 \cr
1 & 3
\end{pmatrix}
\label{eq:2d_quadratic_function}
\end{equation}

Note that equation \\( \eqref{eq:2d_quadratic_function} \\) is a convex quadratic function since the Hessian matrix \\( \mathbf{A} \\) is positive definite. The minimum value is attained at the origin, i.e. \\(\mathbf{x}^{*} = 0\\). Here are some interesting exercises to experiment with:

<ol type="a" class="parens">
  <li>Compute the supremum \(\gamma_{\sup}\) where the iteration goes unstable, then simulate. Is the stability boundary as predicted?</li>
  <li>Compute the optimal step size \(\gamma^*\), then simulate.</li>
  <li>Test gradient descent with \(\gamma\) just above and below the supremum \(\gamma_{\sup}\).</li>
</ol>

Before we answer the above question, it is important to stress out again that the stability of gradient descent depend both on the step size \\(\gamma\\) and the eigenvalues of the Hessian matrix \\( \mathbf{A} \\). Since the matrix \\( \mathbf{A} \\) is given, what we can control is the choice of step-size \\(\gamma\\). 

Recall that the range of stable step size \\(\gamma\\) for gradient descent algorithm obey the following inequality:

\\[
0 < \gamma < \frac{2}{\lambda_{\max}}
\\]

The set of stable step size \\(\gamma\\) form an open interval \\( (0,2/\lambda_{\max}) \\) where 0 is the infimum or greates lower bound (GLB) and \\( 2/\lambda_{\max} \\) is the supremum or least upper bound (LUB). Note that inequality above tell us that step size \\( \gamma \\) has to be positive (there will no improvement with zero step size). It is interesting to look closely at the LUB:

\\[
    \gamma_{\sup} = \frac{2}{\lambda_{\max}} \propto \frac{1}{\lambda_{\max}}
\\]

The eigenvalues \\(\lambda_i\\) of the Hessian matrix \\( \mathbf{A} \\) represent the curvature of the function \\(  f\\) in the eigendirection. Thus, formula above tell us that when the curvature is large, then the range of stable step size \\(\gamma\\) is tight. Vice versa, when the curvature is small, then the range of stable step size \\(\gamma\\) is wide. 

To answer (a), we need to find \\(\lambda_{\max}\\), then calculate \\(\gamma_{\sup}\\). The maximum eigenvalue \\(\lambda_{\max}(\mathbf{A}) \approx 3.618\\) and supremum \\(\gamma_{\sup} \approx 0.5527\\).

{% include figure.html
   id="step_size_sup"
   src="/assets/images/articles/step_size_sup.png"
   alt="Contour plot showing gradient descent trajectories with supremum step sizes"
   caption="Gradient descent trajectories with supremum step size."
%}

<a href="#figure-step_size_sup"></a> illustrates the effect of supremum step size on the gradient descent trajectory. The figure shows that, starting from initial condition, the gradient descent algorithm initially make a large jump into a much lower level set. This is obviously caused by the large step size \\(\gamma_{\sup}\\). What happen next is really interesting. After the first few iteration, gradient descent algorithm cannot move to a lower level set. Instead, it oscillates back and forth around the minimum. Thus, for step size \\(\gamma_{\sup}\\), the gradient descent algorithm shows the behaviour that is on the stability boundary, in the sense that it does not converge nor diverge. This behaviour confirm that the step size \\(\gamma_{\sup}\\) is indeed on the stability boundary, i.e. to have a stable behaviour, the gradient descent algorithm must have step size \\(\gamma < \gamma_{\sup}\\).

{% include figure.html
   id="obj_function_step_size_sup"
   src="/assets/images/articles/obj_function_step_size_sup.png"
   alt="Objective function vs iterations showing gradient descent trajectories with supremum step sizes"
   caption="Objective function vs iterations with supremum step size."
%}

<a href="#figure-obj_function_step_size_sup"></a> is consistent with the level set plot. Initial iteration improve significantly, then the objective value function stuck above the minimum value (precisely at \\(f(x)=2\\)). Again this confirm that the gradient descent algorithm show the behaviour on the stability boundary when we use the step size \\(\gamma_{\sup}\\). This answer (a).

To answer (b) we need to calculate the optimal step size \\(\gamma^\*\\). Recall that the optimal step size \\(\gamma^*\\) of gradient descent algorithm with convex quadratic objective function is given by:

\\[
    \gamma^* = \frac{2}{\lambda_{\min} + \lambda_{\max}}
\\]

Thus, we need to calculate both \\(\lambda_{\min}\\) and \\(\lambda_{\max}\\) of the Hessian matrix \\( \mathbf{A} \\) first, then calculate the optimal step size \\(\gamma^\*\\). The optimal step size \\(\gamma^*=0.4\\).

{% include figure.html
   id="gd-contour"
   src="/assets/images/articles/contour-plot.png"
   alt="Contour plot showing gradient descent trajectories for two different step sizes"
   caption="Gradient descent trajectories with different step sizes. The red trajectory uses the optimal step size, while the blue trajectory uses a smaller step size."
%}

In <a href="#figure-gd-contour"></a> we compare the optimal step-size \\(\gamma^\*=0.4\\) with a smaller step-size \\(\gamma=0.1\\). The level set plot above shows that using the optimal step size \\(\gamma^\*\\) indeed guarantee stability and fast convergence of the gradient descent algorithm. Initially, the iteration take large step toward lower level set since the gradient is large when the point \\(x_k\\) is far from the optimal point \\(x^*=0\\). As \\(x_k\\) gets closer to the optimal point, the gradient gets smaller which cause the iteration to take smaller steps. With smaller step-size \\(\gamma=0.1\\), the gradient descent algorithm also to converge to the minimum but much slower.

{% include figure.html
   id="gd-objfunction-vs-iteration"
   src="/assets/images/articles/gd-second-plot.png"
   alt="Plot showing the objective function's value vs iteration for two different step sizes"
   caption="The objective function's value vs iteration with different step sizes. The red trajectory uses the optimal step size, while the blue trajectory uses a smaller step size."
%}

<a href="#figure-gd-objfunction-vs-iteration"></a> shows how the gradient descent algorithm with optimal step-size \\(\gamma^\*\\) converge to the minimum in only four iterations. This again confirm the optimality of \\(\gamma^*\\). On the other hand, with smaller step size \\(\gamma=0.1\\), the gradient descent algorithm converge to the minimum but about three times slower. This answer (b).

To answer (c), we need to set two step size: one slightly above \\(\gamma_{\sup}\\), and the other one slightly below \\(\gamma_{\sup}\\). Then we run the gradient algorithm to see the effect of step size. Note that \\( \gamma_{\sup} \approx 0.5527 \\). We choose \\( \gamma \approx 0.5627 \\) and \\( \gamma \approx 0.5427 \\).

{% include figure.html
   id="step_size_above_below_sup"
   src="/assets/images/articles/step_size_above_below_sup.png"
   alt="Plot showing the level set plot for two different step sizes above and below the supremum step size."
   caption="The level set plot for two different step sizes above and below the supremum step size."
%}

<a href="#figure-step_size_above_below_sup"></a> shows that when the step size is slightly below \\(\gamma_{\sup}\\), then gradient descent algorithm constantly overshoot the minimum, although eventually converge to it. This oscillation behaviour cause the algorithm to take many iterations to converge to the minimum. This confirms that the gradient descent algorithm is still stable for \\(\gamma < \gamma_{\sup}\\), although the convergence speed is way too slow. We also observe that the gradient descent algorithm is diverging when the step size is slightly above \\(\gamma_{\sup}\\). This confirms that the gradient descent algorithm is unstable when \\(\gamma > \gamma_{\sup}\\).

{% include figure.html
   id="objfunction_above_below_sup"
   src="/assets/images/articles/objfunction_above_below_sup.png"
   alt="Plot showing the objective function vs iterations for two different step sizes above and below the supremum step size."
   caption="The objective function vs iterations with two different step sizes above and below the supremum step size."
%}

<a href="#figure-objfunction_above_below_sup"></a> clearly show the effect of step sizes that are close to the stability boundary \\(\gamma_{\sup}\\). When \\(\gamma < \gamma_{\sup}\\), the gradient descent algorithm converge to the minimum (stable, but slow), while for \\(\gamma > \gamma_{\sup}\\), the gradient descent algorithm diverge (unstable). This answer (c).

#### Experiment 3

In this experiment we will explore how gradient descent algorithm scales with the dimension of an optimization problem. To simplify analysis we will consider quadratic objective function where we know how to calculate both the range of stable step size and optimal step size.

Consider the following quadratic objective function:

\begin{equation}
    f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T \mathbf{Q}\mathbf{x}
\label{eq:quadratic_function}
\end{equation}

where \\(\mathbf{Q} \in \mathbb{R}^{n \times n}\\) is a symmetric positive definite matrix. The optimal point is \\(\mathbf{x}^*=0\\). We choose the initial condition to be \\(\mathbf{x}_0 = [1,1, ..., 1]^T = \mathbf{1}_n\\).

Note that for quadratic objective function \\( \eqref{eq:quadratic_function} \\), the gradient vector is \\(\nabla f(\mathbf{x}) = \mathbf{Q} \mathbf{x}\\), so each iteration computes:

\begin{equation}
    \mathbf{x}_{k+1} = \mathbf{x}_k - \gamma \mathbf{Q} \mathbf{x}_k
\label{eq:gradient_update_quadratic}
\end{equation}

In this experiment, we generate the positive definite matrix \\(\mathbf{Q}=A^T A\\) by first generating random rectangular matrix \\(A \in \mathbb{R}^{m \times n}\\), with \\(m>n\\). We need to be able to generate matrix \\(\mathbf{Q}\\) of various dimension \\(n \times n\\). As we increase the size of \\(n\\), we will investigate how gradient descent scales with the dimension. One way to do this is by custom eigenvalues:

<ol type="a" class="parens">
  <li>Compute the SVD of \(A\): \(A = U \Sigma V^T\)</li>
  <li>Assume that the rank of \(A\) is \(n\), then the eigenvalues \(\lambda_i\) of \(\mathbf{Q}\) will be the square of the non-zero singular values of \(A\), i.e. \(\sigma_i^2\).</li>
  <li>Thus, to design custom eigenvalues \(\hat{\lambda}_i > 0\), we simply replace the corresponding singular values with \(\sqrt{\hat{\lambda}_i}\), resulting in \(\hat{\Sigma}\). Then we construct a new custom \(\hat{A}\) and \(\hat{\mathbf{Q}}\):
    \begin{equation}
      \hat{A} = U \hat{\Sigma} V^T
      \quad \Longrightarrow \quad
      \hat{\mathbf{Q}} = \hat{A}^T \hat{A}
    \end{equation}
  </li>
</ol>

We choose the eigenvalues of \\( \mathbf{Q} \\) in the range \\(1 \leq \lambda_i(\mathbf{Q}) \leq 10\\) and \\(n \in \\{10,100,500,1000,2000\\}\\). To measure convergence we use relative error criteria \\(\mathbf{e}_k = \\| \mathbf{x}_k - \mathbf{x}^* \\| / \\| \mathbf{x}_0 - \mathbf{x}^* \\| \leq 10^{-3}\\). Also for safe guard, we set maximum iterations to 10,000 steps. Since runtime can be varied, we run the gradient descent 30 times for each value of \\(n\\), then calculate median of runtime.

For all \\(n \in \\{10,100,500,1000,2000\\}\\), the gradient descent algorithm with \\( \gamma=0.01 \\) succeed to converge under the relative error criteria.

{% include figure.html
   id="iter-vs-dim"
   src="/assets/images/articles/iter-vs-dim.png"
   alt="Plot showing number of iterations to converge vs the problem dimension."
   caption="Number of iterations to converge vs the problem dimension."
%}

<a href="#figure-iter-vs-dim"></a> above shows that the number of iterations to converge varies as the dimension \\(n\\) grows. Initially, the iterations decreases from 520 to 445 as the dimension increase from \\(n=10\\) to \\(n=100\\). But as the dimension grows even larger from \\(n=10\\) to \\(n=2000\\), the iterations does not exhibit clear increasing trend. Instead, the iterations fluctuate within a narrow range between 445 to 471. This suggest that *increase in dimension does not necessarily cause significant increase in gradient descent iterations* required to reach prescribed relative error tolerance. In particular, this experiment suggest that iterations are relatively stable for large dimension.

The result can be explained from the theoretical analysis of convergence rate. Recall from equation \\( \eqref{eq:gradient_update_quadratic} \\) that the gradient descent iteration is given by \\(\mathbf{x}\_{k+1} = (I-\gamma \mathbf{Q}) \mathbf{x}\_k\\), where the contraction matrix is \\((I-\gamma \mathbf{Q}) \\). Note that the convergence rate is determined by the slowest-decaying eigenvalue of the contraction matrix, i.e the spectral radius of \\((I-\gamma \mathbf{Q})\\), which is the largest absolute eigenvalue \\(\rho (I-\gamma \mathbf{Q}) = \max_i \|1-\gamma \lambda_i\|\\). We can see that *the spectral radius does not depend on the dimension \\(n\\), but instead depend on the minimum and maximum eigenvalue of \\(\mathbf{Q}\\)* (step size is fixed), as shown by equation \\( \eqref{eq:spectral_radius_minmax_eigenvalue} \\). And since in this experiment we set the eigenvalue range fixed from \\(\lambda_{\min}=1\\) to \\(\lambda_{\max}=10\\), then *the spectral radius is the same for various value of dimension \\(n\\)*. This explain why the number of iterations needed to achieve convergence is relatively stable within a narrow range.

Why for \\(n=10\\) the number of iterations is so different? Although the spectrum of eigenvalues is identical for different \\(n\\), the projection of initial condition \\(\mathbf{x}_0=\mathbf{1}_n\\) for \\(n=10\\) might have a very different component along the slowest-decaying eigenvector compared to the other values of \\(n\\). This serves as a plausible explanation for this particular result.

{% include figure.html
   id="runtime-vs-dim"
   src="/assets/images/articles/runtime-vs-dim.png"
   alt="Plot showing runtime (seconds) to converge vs the problem dimension."
   caption="Runtime (seconds) to converge vs the problem dimension."
%}

<a href="#figure-runtime-vs-dim"></a> above shows a log-log plot of median runtime vs dimension \\(n\\) for different problem dimensions, based on 30 runs for each dimension. We choose the median since it is less sensitive to runtime variability caused by interruptions, background processes, or other system activity.

This result suggests that runtime grows quite rapidly as dimension \\(n\\) increases, as expected. For \\(n=10\\) and \\(n=100\\), the median runtimes are nearly identical. One possible explanation is *fixed overhead for relatively small dimensions*, which is a baseline computational cost that does not scale with dimension. *As \\(n\\) grows larger, the dimension-dependent computation (large matrix-vector multiplication) dominates*, and runtime grows.

Recall from equation \\( \eqref{eq:gradient_update_quadratic} \\) that the gradient descent update is given by \\(\mathbf{x}_{k+1} = \mathbf{x}_k - \gamma \mathbf{Q} \mathbf{x}_k\\). For large matrices, the runtime is dominated by matrix-vector multiplication \\(\mathbf{Q} \mathbf{x}\\), where the computational cost is \\(O(n^2)\\). Thus, the work per iteration increases quadratically with the problem dimension \\(n\\).

Since total work \\(\approx\\) work per iteration \\(\times\\) number of iterations, and the number of iterations in the previous experiment is relatively stable for large dimensions, total work (and thus runtime) is also expected to increase approximately quadratically with \\(n\\). This provides a theoretical explanation for the sharp increase in runtime observed at large dimensions.

For \\(n=100\\) to \\(n=2000\\), we fit a scaling law where \\(T(n)=Cn^p\\), with \\(p=1.73\\). The empirical scaling does not follow the theoretical quadratic scaling, which may be caused by factors such as measurement variability, implementation-dependent costs, and others.

{% include figure.html
   id="runtime-vs-dim"
   src="/assets/images/articles/obj-value-vs-iterations.png"
   alt="Plot showing objective function's value vs iterations."
   caption="Runtime (seconds) to converge vs the problem dimension."
%}

<a href="#figure-runtime-vs-dim"></a> above shows semilog plot of objective function's value versus iterations of gradient descent for different problem dimension. The result clearly show that the iteration undergo transient phase (curved line) starting from the initial condition. This is cause by combined contributions of *multiple eigenvalue modes that decay at different rate*. 

As the iterations progress, *faster-decaying modes become negligible and slower-decaying modes become dominant*. Eventually, the decrease of the objective function's value obey geometric decay, which in semilog plot, resulting into a straight line.

Recall that since \\( f(\mathbf{x}^*)=0 \\), the objective function behaviour can be approximated by \\(f(x_k) \approx C\rho^k\\), with \\(C>0\\) and \\(0<\rho<1\\). Take the log on both sides:

\\[
   \log f(x_k) \approx \log C + k \log \rho
\\]

This has the simple form \\(y = b + mk\\), which is a linear function of the iteration \\(k\\), where the intercept is \\(b=\log C\\) and the slope is \\(m=\log \rho < 0\\). Thus, *the objective function's value decreases linearly in a semilog plot*, as shown in the figure above, which is consistent with geometric decay of the gradient descent algorithm.

Another interesting result in the plot is that as the dimension \\(n\\) increases, the iterations stop at a higher objective function value. This can be explained by the relative error criterion used to stop the iterations. Recall the relative error criterion:

\\[
   \frac{\\| \mathbf{x}_k - \mathbf{x}^\* \\|_2}{\\| \mathbf{x}_0 - \mathbf{x}^\* \\|_2}
   \leq \varepsilon
\\]

In our experiment, \\(\varepsilon=10^{-3}\\), the optimal point is \\(\mathbf{x}^\* = \mathbf{0}_n\\), and the initial point is \\(\mathbf{x}_0 = \mathbf{1}_n\\). Thus, at iteration \\(k\\), we have:

\\[
   \\| \mathbf{x}_k \\|_2 \leq 10^{-3} \sqrt{n}
\\]

The formula tells us that as \\(n\\) increases, the upper bound on the norm of \\(\mathbf{x}_k\\) also increases proportionally to \\(\sqrt{n}\\). This leads to a higher objective function value \\(f(\mathbf{x}_k)\\), as shown in the plot.

## 3. Heavy Ball

Consider two-dimensional objective function with saddle point:

\begin{equation}
    f(x) = 15 e^{-0.25(x+2)^2} + 0.5 e^{-2(x-2)^2} + x^2 + y^2
\label{eq:saddle_function}
\end{equation}

The global minimum is attained at \\((x^\*,y^\*)=(1.0181,0)\\), where the minimum value of the objective function is \\(f(x^\*,y^*) \approx 2.6477\\).

{% include figure.html
   id="saddle-function"
   src="/assets/images/articles/saddle-function.png"
   alt="Figure showing objective function with saddle point."
   caption="Objective function with saddle point."
%}

<a href="#figure-saddle-function"></a> shows 3D plot of the objective function. There are two minima: one of them is local and the other one is global. Between the two minima, there exist a saddle point where the curvature is positive in one direction and negative in the other orthogonal direction.

Let's choose initial point \\( (-5,3) \\) and step size \\( \gamma=0.1\\), then apply gradient descent to objective function \\( \eqref{eq:saddle_function} \\).

{% include figure.html
   id="gd-saddle-point"
   src="/assets/images/articles/gd-saddle-point.png"
   alt="Figure showing gradient descent trajectories on level set of objective function with saddle point."
   caption="Gradient descent trajectories on level set of objective function with saddle point."
%}

<a href="#figure-gd-saddle-point"></a> shows that starting from initial condition \\( (-5,3) \\), the gradient descent algorithm struggles near the saddle point, as the gradient at the saddle point nearly vanishes, causing the iteration converge painfully slow to the global minimum.

{% include figure.html
   id="objfunction-saddle-point"
   src="/assets/images/articles/objfunction-saddle-point.png"
   alt="Figure showing objective function's value have difficulty to decrease near saddle point."
   caption="Objective function's value have difficulty to decrease near saddle point."
%}

 We can see from <a href="#figure-objfunction-saddle-point"></a> where gradient descent algorithm have difficulty to escape from saddle point since the gradient there may be negligibly small. Recall that gradient descent algorithm is based on current gradient to take the next step, thus small gradient around saddle point result in small step for the next iteration. This pose serious problem for the gradient descent algorithm as it struggle to go past through saddle point to find the minimum.

The classical momentum method or *heavy ball* address this issue by integrates past gradient and thus provide momentum to the optimization algorithm such that it can go pass through saddle point of small gradient and converge toward the minimum.

### 3.1. The Momentum Method

Through a simple numerical experiment, we have seen how gradient descent struggles near or around saddle point. Recall the update rule of gradient descent:

\\[
  \mathbf{x}\_{k+1} =  \mathbf{x}\_{k} - \gamma \nabla f(\mathbf{x}_{k})
\\]

Note that gradient descent uses current gradient value to take the next step. However, gradient near saddle point may be negligibly small, which cause serious difficulty for gradient descent to go pass through saddle point.

The classic momentum method of *heavy ball* address this issue by integrate past gradient information and hence provide *inertia* to the algorithm to carry it through the flat regions of saddle point. 

This simple idea is similar to rolling down a a massive ball down the objective landscape. We can image a scenario where the ball initially start at region with high gradient, then the ball gain momentum and it will have inertia to roll through flat saddle regions.

The heavy ball algorithm modify the update rule of gradient descent by adding momentum. Heavy ball algorithm has two update rule. Informally, they can be written as follow:

\begin{aligned}
  \text{New position} &= \text{Old position} + (\text{Step size})(\text{Downhill force}) + (\text{Decay rate})(\text{Old velocity}) \cr
  \text{New velocity} &= (\text{Decay rate})(\text{Old velocity}) + (\text{Step size})(\text{Downhill force})
\end{aligned}

So basically there are two update rule: an update for position and another update for velocity. Suppose the ball is near flat saddle regions where the downhill force is virtually zero. We can still have significant update for the position since we have contribution from nonzero *velocity*, as shown by the first update rule. The second update rule shows how the velocity at each step is updated using the contribution from downhill force. Thus, past gradient contribute to the latest velocity. This mechanisms makes the heavy ball algorithm can roll through flat saddle regions. The price to pay is a new hyperparameter \\( 0 < \beta < 1 \\), the *decay rate* for the velocity update.

Mathematically, the heavy ball update rule is given by:

\begin{aligned}
  \mathbf{x}\_{k+1} &=  \mathbf{x}\_{k} - \gamma \nabla f(\mathbf{x}\_{k}) + \beta \mathbf{v}\_k \cr
  \mathbf{v}\_{k+1} &= \beta \mathbf{v}\_k - \gamma \nabla f(\mathbf{x}_{k})
\end{aligned}

which can be rewritten more compactly as follow:

\begin{equation}
\begin{aligned}
  \mathbf{v}\_{k+1} &= \beta \mathbf{v}\_k - \gamma \nabla f(\mathbf{x}_{k}) \cr
  \mathbf{x}\_{k+1} &=  \mathbf{x}\_{k} + \mathbf{v}\_{k+1}
\end{aligned}
\label{eq:hb_update_rule}
\end{equation}

Starting with initial velocity \\( \mathbf{v}_0 \\), we can use the update rule for velocity \\( \eqref{eq:hb_update_rule} \\) and work recursively (current output becomes the next input) to get the expression for \\( \mathbf{v}_k \\):

\begin{equation}
  \mathbf{v}\_k = \beta^k \mathbf{v}\_0 - \gamma \left( \beta^{k-1} g_0 + \beta^{k-2} g_1 + \ldots + \beta g_{k-2} + g_{k-1} \right)
\label{eq:integrate_past_gradient}
\end{equation}

where we use \\( g_k = \nabla f(\mathbf{x}_k) \\) for simplicity. Note that *older gradient receive progressively smaller weights*, since \\( 1 > \beta > \beta^2 > \ldots > \beta^{k-1} > \beta^k\\). Thus velocity \\( \mathbf{v}_k \\) is *exponentially discounted sum of past gradient*, where influence of gradient decreases exponentially as it gets older. If \\( \beta \approx 0 \\), then we forget old gradient quickly. If \\( \beta \approx 1 \\), we remember old gradient longer.

The algorithm of heavy ball is described as follow:

<table style="width:100%; border-collapse:collapse; border-top:2px solid black; border-bottom:2px solid black; font-size:1.05em;">

  <thead>
    <tr style="border-bottom:2px solid black;">
      <th colspan="2" style="text-align:left; padding:4px 8px; font-weight:normal;">
        <strong>Algorithm 2</strong> &nbsp; Heavy Ball Method
      </th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td style="width:45px; padding:3px 10px; vertical-align:top;">1:</td>
      <td style="padding:3px 8px;"><strong>Input</strong>: Initial iterate \(\mathbf{x}_0\) and \(\mathbf{v}_0\), maximum iteration \( N \), step size \( \gamma \) and decay rate \( \beta \).</td>
    </tr>

    <tr>
      <td style="padding:3px 10px; vertical-align:top;">2:</td>
      <td style="padding:3px 8px;"><strong>for</strong> \(k=1,2,\ldots\,N\) <strong>do</strong></td>
    </tr>

    <tr>
      <td style="padding:3px 10px; vertical-align:top;">4:</td>
      <td style="padding:3px 8px 3px 35px;">
        Compute gradient of the objective function \(g_k = \nabla f(\mathbf{x}_k) \).
      </td>
    </tr>

<tr>
      <td style="padding:3px 10px; vertical-align:top;">6:</td>
      <td style="padding:3px 8px 3px 35px;">
        Set the new velocity as
        \(\mathbf{v}_{k+1}\leftarrow \beta \mathbf{v}_k-\gamma g_k \).
      </td>
    </tr>

    <tr>
      <td style="padding:3px 10px; vertical-align:top;">6:</td>
      <td style="padding:3px 8px 3px 35px;">
        Set the new position as
        \(\mathbf{x}_{k+1}\leftarrow \mathbf{x}_k + \mathbf{v}_{k+1} \).
      </td>
    </tr>

    <tr>
      <td style="padding:3px 10px; vertical-align:top;">7:</td>
      <td style="padding:3px 8px;"><strong>end for</strong></td>
    </tr>
  </tbody>
</table>

### 3.2. Numerical Experiment

In this subsection we will test and experiment with heavy ball algorithm on several objective function so that we can see how this algorithm use inertia to overcome difficulty experienced by gradient descent.

#### Experiment 1

Recall the two-dimensional objective function with saddle region \\( \eqref{eq:saddle_function} \\):

\begin{equation}
    f(x) = 15 e^{-0.25(x+2)^2} + 0.5 e^{-2(x-2)^2} + x^2 + y^2
\end{equation}

The global minimum is attained at \\((x^\*,y^\*)=(1.0181,0)\\), where the minimum value of the objective function is \\(f(x^\*,y^*) \approx 2.6477\\). Now we will implement heavy ball algorithm to observe its behavior around the saddle region. We choose the same initial point \\( (-5,3) \\) and step size \\( \gamma=0.1 \\). Specific for the heavy ball algorithm, we choose another hyperparameter decay rate \\( \beta = 0.6 \\) and initial velocity \\( \mathbf{v}_0  = 0\\).

{% include figure.html
   id="hb_on_saddle"
   src="/assets/images/articles/hb_on_saddle.png"
   alt="Figure showing comparison of gradient descent and heavy ball trajectories on the level set of objective function with saddle region."
   caption="Comparison of gradient descent (blue line) and heavy ball (red line) trajectories on the level set of objective function with saddle region."
%}

 <a href="#figure-hb_on_saddle"></a> above shows both gradient descent and heavy ball iteration for comparison. It can be seen clearly that the heavy ball algorithm outperform gradient descent with the same step size. While gradient descent algorithm struggle near saddle point, the heavy ball algorithm can pass through saddle point quite easily. By integrating past gradients, the heavy ball iteration have enough momentum to go past through saddle region although saddle region have small gradient.

 {% include figure.html
   id="hb-objfunction-saddle"
   src="/assets/images/articles/hb-objfunction-saddle.png"
   alt="Figure showing comparison of gradient descent and heavy ball on the objective function's value vs iterationn."
   caption="Comparison of gradient descent (blue line) and heavy ball (red line) on the objective function's value vs iteration."
%}

 <a href="#figure-hb-objfunction-saddle"></a> above shows that the heavy ball algorithm converge to the minimum much faster compared to the gradient descent algorithm for the same step size. While the heavy ball already reach the minimum, gradient descent still struggle around the saddle point. This stark contrast between heavy ball and gradient descent in terms of ability to go pass through saddle point demonstrate the eminence of integrating past gradients instead of just using current gradient.

By choosing decay rate \\(\beta=0.6\\), this provide the heavy ball algorithm just enough momentum to roll through flat saddle region while at the same maintain good convergence behaviour toward the minimum.

#### Experiment 2

Previously, we already see how gradient descent algorithm struggle near saddle point. This problem is addressed by the heavy ball algorithm that integrates past gradients and hence provide momentum such that the iteration can move pass through saddle point that have vanishingly small gradient.

In this exercise we will experiment more with the heavy ball algorithm. Consider the following objective function:

\begin{equation}
    f(x) = \sin^2(x) + 0.1 x^2
\label{eq:nonconvex_objfunction}
\end{equation}

{% include figure.html
   id="nonconvex-objfunction"
   src="/assets/images/articles/nonconvex-objfunction.png"
   alt="Figure showing nonconvex objective function with multiple local minimum and one unique global minimum."
   caption="Nonconvex objective function with multiple local minimum and one unique global minimum."
%}

  The objective function as shown by <a href="#figure-nonconvex-objfunction"></a> is obviously non-convex, where there are several local minimum and one unique global minimum. The global minimum is attained at \\(x^\*=0\\) where \\(f(x^*)=0\\). We choose three initial points: \\( x=1 \\) which is close to global minimum, also \\( x=2 \\) and \\( x=5 \\) where both are close to local minimum. We apply first gradient descent to objective function \\( \eqref{eq:nonconvex_objfunction} \\) with step size \\( \gamma=0.3 \\).

  {% include figure.html
   id="gd-on-nonconvex"
   src="/assets/images/articles/gd-on-nonconvex.png"
   alt="Figure showing gradient descent applied to nonconvex objective function."
   caption="Gradient descent applied to nonconvex objective function."
%}

<a href="#figure-gd-on-nonconvex"></a> above shows that when applying gradient descent algorithm to nonconvex objective funtion, then the result will strongly depend on the initial point. The gradient descent algorithm have tendency to converge to the nearest local minimum, hence it will converge to global minimum only when the initial point is sufficiently close to the global minimum.

Will heavy ball algorithm with momentum be able to escape from local minimum to converge to the global minimum ? Let's put it to test.

What's interesting is the following experiment: if we set the initial condition \\(x_0=5\\), then negative gradient will always point to the left (as shown by <a href="#figure-nonconvex-objfunction"></a>) until we reach the local minimum where the gradient is zero. Can the heavy ball algorithm gain enough momentum along this path to escape the local minimum and then  reach the global minimum ? Let's conduct the experiment.

{% include figure.html
   id="hb-on-nonconvex"
   src="/assets/images/articles/hb-on-nonconvex.png"
   alt="Figure showing heavy ball applied to nonconvex objective function."
   caption="Heavy ball with various decay rate \( \beta 
   \) applied to nonconvex objective function."
%}

<a href="#figure-hb-on-nonconvex"></a> above shows some very interesting behaviour of heavy ball algorithm. When the initial point is close to the global minimum (\\(x_0=1 \\)), the heavy ball iteration converge to the global minimum with a bit of oscillation caused by the inherent momentum of iterate \\(x_k\\). 

More interesting behaviour is shown when the algorithm start far from the global minimum. In this case, to reach the global minimum, the iterate \\(x_k\\) must go pass through the local minimum. If we set small decay rate (\\(\beta=0.6\\)), the blue line show that the iteration stuck at the local minimum. If we set higher decay rate (\\(\beta=0.9\\)), the purple line show us that the iterate gain too much momentum it pass through the local minimum, then it gain even more momentum such that it passes through the global minimum, climb up and then stuck again at the next local minimum.

Finally, when the decay rate is good enough (\\(\beta=0.7\\)) as shown by the green line, the iteration have enough momentum to passes through the local minimum, then converge with a bit of oscillation to the global minimum.

This experiment shows that unlike gradient descent, the heavy ball algorithm depend less to initial condition, but depend more on the value of decay rate \\(\beta\\). Larger decay rate give more momentum, and vice versa.

## 4. Conclusion

### 4.1. Gradient Descent

In [Gradient Descent](#2-gradient-descent), we have explored the mathematical foundations and computational behavior of gradient descent. Starting from the principle of steepest descent, we derived the gradient descent update rule and examined how the choice of step size determines the stability and convergence behavior of the algorithm.

Our mathematical analysis shows that for a convex quadratic objective function with a positive definite Hessian matrix, the stability of gradient descent is determined by the eigenvalues of the Hessian matrix. In particular, the maximum eigenvalue determines the upper bound on the stable step size, while the minimum and maximum eigenvalues jointly determine the optimal step size for fastest convergence. Furthermore, the convergence analysis of strongly convex functions reveals that gradient descent exhibits linear convergence, with its convergence rate governed by the condition number of the Hessian matrix. These results highlight an important insight: the geometry of the objective function plays a fundamental role in determining the convergence behavior of gradient descent.

The numerical experiments further illustrate these theoretical results. We observed that an excessively large step size can cause oscillation or divergence, whereas a sufficiently small step size guarantees stability but may lead to slow convergence. Choosing an appropriate step size can significantly accelerate convergence. Our experiments also demonstrate that gradient descent struggles when approaching saddle regions, where the gradient becomes negligibly small.

Finally, the scalability experiment reveals an interesting distinction between the number of iterations and computational runtime. When the eigenvalue spectrum is kept fixed, increasing the problem dimension does not necessarily increase the number of iterations required to achieve a prescribed relative error tolerance. However, the computational cost per iteration increases with dimension due to the matrix-vector multiplication, resulting in a substantial increase in overall runtime.

### 4.2. Heavy Ball

In [Heavy Ball](#3-heavy-ball), we explored the classical momentum method, known as the heavy ball algorithm, as an extension of gradient descent. Unlike gradient descent, which relies exclusively on the current gradient to determine the next step, heavy ball incorporates information from previous gradients through a velocity variable. This introduces inertia into the optimization process, allowing the algorithm to maintain its movement even in regions where the gradient is small.

Our numerical experiments demonstrate the advantage of this mechanism when navigating saddle regions. While gradient descent struggles to make progress near a saddle point due to the vanishingly small gradient, heavy ball can exploit accumulated momentum to move through the region and continue toward the minimum. This illustrates how incorporating information from previous iterations can improve the behavior of an optimization algorithm.

However, momentum does not automatically guarantee better convergence. Our experiments on a non-convex objective function with multiple local minima reveal that the behavior of heavy ball depends strongly on the choice of decay rate. A relatively small decay rate may provide insufficient momentum to escape a local minimum, whereas an excessively large decay rate can produce substantial oscillations and cause the iteration to overshoot minima. With an appropriate decay rate, however, heavy ball can acquire sufficient momentum to pass through a local minimum and converge toward the global minimum in our experiment.

These observations highlight an important trade-off in momentum-based optimization: momentum can help overcome flat regions and local minima, but excessive momentum can also introduce oscillatory behavior and destabilize convergence. Therefore, selecting an appropriate decay rate is essential to obtaining desirable behavior from the heavy ball algorithm.

{% include reproducibility.html
   description="The numerical experiments and figures presented in this article were generated using Python. The complete implementation and experimental setup are available in the accompanying Jupyter notebook."
   url="https://github.com/wap314/Gradient-Descent-Experiment/blob/main/GD_experiment.ipynb"
%}