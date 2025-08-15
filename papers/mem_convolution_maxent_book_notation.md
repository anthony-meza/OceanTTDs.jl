
# Maximum Entropy with Convolution Constraints — Book-Style Notation

This note derives the Maximum Entropy (MaxEnt) solution when the constraints are **linear convolutions**
of an unknown distribution with a known kernel, using the **notation found in many imaging texts**:
features \(S_j(s,t)\), multipliers \(\lambda_j\), normalization \(Z(\lambda)\), and a base measure \(\mu(s,t)\).
Both **noise-free** (hard) and **Gaussian-noisy** (soft) constraints are treated.

---

## Notation and Setting

- Discrete 2-D grid: indices \((s,t) \in \mathcal I \subset \mathbb{Z}^{n,m}\).
- Unknown, nonnegative, normalized image/distribution:
  \[
  P(s,t) \ge 0,\qquad \sum_{(s,t)\in \mathcal I} P(s,t) = 1.
  \]
- Base measure (reference image/weights): \(\mu(s,t) > 0\) on \(\mathcal I\).
- A collection of **linear features** \(\{S_j(s,t)\}_{j=1}^m\). In convolutional imaging,
  a common choice is a **shifted kernel** (e.g., point-spread function, PSF)
  \[
  S_j(s,t) = H(s - s_j,\; t - t_j),
  \]
  so that the feature expectation is a **sample of a convolution**:
  \[
  \sum_{s,t} S_j(s,t)\,P(s,t) = (H * P)[s_j,t_j].
  \]
- Observed (or target) values \(y_j\) for \(j=1,\dots,m\).
  - **Hard constraints (noise-free):** \(\sum_{s,t} S_j(s,t)\,P(s,t) = y_j\).
  - **Gaussian-noisy constraints:** \(y \mid P \sim \mathcal N\big( \mathbb E_P[S],\,\Sigma \big)\), where
    \(\big[\mathbb E_P[S]\big]_j := \sum_{s,t} S_j(s,t) P(s,t)\) and \(\Sigma \succ 0\) is the noise covariance.

---

## MaxEnt Objective

We maximize (relative) entropy subject to constraints. Equivalently, minimize the KL divergence
\(\mathrm{KL}(P\|\mu) = \sum_{s,t} P(s,t)\log \frac{P(s,t)}{\mu(s,t)}\).

### (A) Noise-free (hard) constraints
\[
\begin{aligned}
\min_{P \ge 0,\;\sum P=1}
&\quad \sum_{s,t} P(s,t)\,\log\!\frac{P(s,t)}{\mu(s,t)} \\
\text{s.t.}
&\quad \sum_{s,t} S_j(s,t)\,P(s,t) = y_j,\qquad j=1,\dots,m.
\end{aligned}
\]

### (B) Gaussian-noisy (soft) constraints
\[
\min_{P \ge 0,\;\sum P=1}
\quad \sum_{s,t} P(s,t)\,\log\!\frac{P(s,t)}{\mu(s,t)}
+\frac{1}{2}\big(\mathbb E_P[S]-y\big)^\top \Sigma^{-1}\big(\mathbb E_P[S]-y\big).
\]

---

## Derivation — Hard Constraints (noise-free)

Form the Lagrangian with multipliers \(\alpha\) (for normalization) and \(\{\lambda_j\}\) (one per constraint):
\[
\mathcal L(P,\alpha,\lambda) = -\sum_{s,t} P(s,t)\,\log\!\frac{P(s,t)}{\mu(s,t)}\;
+ \alpha\!\left(\sum_{s,t} P(s,t)-1\right) + \sum_{j=1}^m \lambda_j
\left(\sum_{s,t} S_j(s,t)\,P(s,t) - y_j\right).
\]

First-order stationarity for interior \(P(s,t)>0\):
\[
\frac{\partial \mathcal L}{\partial P(s,t)} = -\!\left(\log\frac{P(s,t)}{\mu(s,t)} + 1\right)
+ \alpha + \sum_{j=1}^m \lambda_j S_j(s,t) = 0.
\]

Solve for \(P\):
\[
\log\frac{P(s,t)}{\mu(s,t)} = \alpha - 1 + \sum_{j=1}^m \lambda_j S_j(s,t)
\quad\Longrightarrow\quad
P(s,t) = c\, \mu(s,t)\,\exp\!\left(\sum_{j=1}^m \lambda_j S_j(s,t)\right),
\]
with \(c=e^{\alpha-1}\). Enforce normalization to find the **partition function**
\[
Z(\lambda) := \sum_{(s,t)\in\mathcal I} \mu(s,t)\,
\exp\!\left(\sum_{j=1}^m \lambda_j S_j(s,t)\right),
\quad c=\frac{1}{Z(\lambda)}.
\]

**MaxEnt solution (hard constraints):**
\[
\boxed{\;
P_\lambda(s,t) = \frac{\mu(s,t)\,\exp\!\left(\sum_{j=1}^m \lambda_j S_j(s,t)\right)}
{Z(\lambda)}\;}, \qquad
Z(\lambda)=\sum_{s,t} \mu(s,t)\,e^{\sum_j \lambda_j S_j(s,t)}.
\]

The multipliers \(\lambda=\{\lambda_j\}\) are chosen so that the constraints hold:
\[
y_j = \sum_{s,t} S_j(s,t)\,P_\lambda(s,t) \;=\; \frac{\partial}{\partial \lambda_j}\log Z(\lambda),
\quad j=1,\dots,m.
\]

> **Interpretation:** \( \log Z(\lambda) \) is the cumulant-generating function; its gradient w.r.t. \(\lambda_j\) equals the model expectation of \(S_j\).

---

## Derivation — Gaussian-Noise (soft) constraints

Use the identity (Fenchel conjugate of a quadratic), for \(\Sigma \succ 0\) and any \(r\):
\[
\frac{1}{2}\,r^\top \Sigma^{-1} r = \max_{\eta\in\mathbb{R}^m}\left\{\eta^\top r - \frac{1}{2}\,\eta^\top \Sigma\,\eta\right\}.
\]
Set \(r = \mathbb E_P[S] - y\). The soft problem becomes a saddle problem:
\[
\min_{P \in \Delta}\max_{\eta}\;\sum_{s,t} P\log\frac{P}{\mu}
+ \eta^\top\!\big(\mathbb E_P[S]-y\big) - \tfrac{1}{2}\eta^\top\Sigma \eta.
\]

Swap min/max (convex–concave), and minimize over \(P\) first. This is the **same inner problem**
as above with \(\lambda \leftarrow \eta\), yielding again
\[
P_\eta(s,t) = \frac{\mu(s,t)\,\exp\!\left(\sum_{j=1}^m \eta_j S_j(s,t)\right)}{Z(\eta)},
\quad Z(\eta)=\sum_{s,t}\mu(s,t)\,e^{\sum_j \eta_j S_j(s,t)}.
\]

Plugging back gives the **dual (now a minimization in \(\lambda\equiv\eta\))**:
\[
\boxed{\;
\phi(\lambda) = \log Z(\lambda)\;-\;\lambda^\top y\;+\;\tfrac{1}{2}\,\lambda^\top \Sigma\,\lambda
\quad\text{(minimize over }\lambda\in\mathbb{R}^m\text{)}.
\;}
\]

- **Gradient and Hessian:**
\[
\nabla \phi(\lambda) = \mathbb E_{P_\lambda}[S]\;-\;y\;+\;\Sigma\,\lambda,
\qquad
\nabla^2 \phi(\lambda) = \underbrace{\operatorname{Cov}_{P_\lambda}(S,S)}_{\succeq 0} + \Sigma \;\succeq\; 0.
\]
Hence \(\phi\) is convex (and **strongly** convex if \(\Sigma\succ 0\)), so numerical minimization in \(\lambda\) is well-behaved.

- **Noise-free limit:** as \(\Sigma\to 0\), \(\phi(\lambda)\to \log Z(\lambda) - \lambda^\top y\), whose stationarity enforces
  \(\mathbb E_{P_\lambda}[S]=y\) exactly.

---

## Convolution Specialization

If each feature is a **shifted kernel** \(S_j(s,t)=H(s-s_j,\,t-t_j)\), then
\[
\mathbb E_{P}[S_j] = \sum_{s,t} H(s-s_j,\,t-t_j)\,P(s,t) = (H * P)[s_j,t_j].
\]
Equivalently, if you **vectorize** \(P\) over \((s,t)\), write a matrix \(A\) with rows
\(A_{j,(s,t)} = S_j(s,t)\), then
\[
\mathbb E_{P}[S] = A\,\mathrm{vec}(P),\qquad
\sum_{j}\lambda_j S_j(s,t) = \big[A^\top \lambda\big]_{(s,t)}.
\]
The MaxEnt solution still has the exponential-family form
\[
P_\lambda(s,t) \propto \mu(s,t)\,\exp\!\Big(\big[A^\top \lambda\big]_{(s,t)}\Big).
\]

> In 1-D convolution with kernel \(h\), \(A^\top \lambda\) is the **cross-correlation** of \(\lambda\) with \(h\) (i.e., convolution with the reversed kernel).

---

## Practical Algorithm (soft constraints)

1. **Given:** \(\mu(s,t)\), features \(S_j(s,t)\) (or convolution kernel \(H\)), observed \(y\), noise covariance \(\Sigma\).
2. **Define** \(Z(\lambda)=\sum_{s,t}\mu(s,t)\,\exp(\sum_j \lambda_j S_j(s,t))\) and
   \(P_\lambda(s,t)=\mu(s,t)\,\exp(\sum_j \lambda_j S_j(s,t))/Z(\lambda)\).
3. **Minimize** \(\phi(\lambda)=\log Z(\lambda)-\lambda^\top y+\tfrac12 \lambda^\top \Sigma \lambda\) by L-BFGS or Newton–CG.
   - Use **log-sum-exp** to compute \(\log Z\) stably.
   - For convolution, compute \(A^\top \lambda\) and \(A P_\lambda\) with **(cross-)convolutions** (FFT for speed).
4. **Return** \(P^\star=P_{\lambda^\star}\). Check residuals: if \(\Sigma=\sigma^2I\), then
   \(\| \mathbb E_{P^\star}[S] - y \|^2 / (m \sigma^2) \approx 1\) if noise is correctly scaled.

---

## Summary of Key Formulas

- **Exponential-family (book form):**
\[
P_\lambda(s,t) = \frac{\mu(s,t)\,\exp\!\Big(\sum_{j=1}^m \lambda_j\,S_j(s,t)\Big)}{Z(\lambda)},\quad
Z(\lambda) = \sum_{s,t} \mu(s,t)\,\exp\!\Big(\sum_{j=1}^m \lambda_j\,S_j(s,t)\Big).
\]

- **Hard constraints:** choose \(\lambda\) so that \(\sum_{s,t} S_j(s,t)\,P_\lambda(s,t)=y_j\).
- **Soft (Gaussian) constraints:** minimize
\[
\phi(\lambda)=\log Z(\lambda)-\lambda^\top y+\tfrac12\,\lambda^\top \Sigma\,\lambda,
\quad \nabla\phi(\lambda)=\mathbb E_{P_\lambda}[S]-y+\Sigma\lambda.
\]

- **Convolution:** \(S_j(s,t)=H(s-s_j,\,t-t_j)\Rightarrow \mathbb E_P[S_j]=(H*P)[s_j,t_j].\)

---

### Notes
- Sign conventions vary across texts. If an alternative derivation yields \(P \propto \mu \exp(-\sum_j \tilde\lambda_j S_j)\),
  simply redefine \(\lambda_j := -\tilde\lambda_j\) to match the book form above.
- On non-uniform grids, embed cell areas into \(\mu(s,t)\).
- If constraints are not independent, MaxEnt still yields a unique \(P_\lambda\); only the span of \(\{S_j\}\) is controlled by data, the remainder by \(\mu\).

