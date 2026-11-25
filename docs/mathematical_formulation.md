# Advantage uncertainty and allocation

Write $z_i$ for a log-speedup estimate and $r_i=f(z_i)$ for its reward. Correctness failures carry a configurable fixed penalty into the group and have zero measurement variance. Sequence padding is masked independently of correctness. For an active group of $n>1$, set $P=I-\mathbf{1}\mathbf{1}^{T}/n$, $c=Pr$, $d=n-\mathrm{ddof}$, and $s=\sqrt{c^{T}c/d}$.

For GRPO with denominator $s+\epsilon$,

$$
A(r)=\frac{c}{s+\epsilon},\qquad
J=\frac{P}{s+\epsilon}-\frac{cc^T}{d\,s(s+\epsilon)^2}.
$$

This expression applies for $s>0$ in the normalized branch. The selected single-turn Dr. Kernel convention uses $\mathrm{ddof}=1$ and $\epsilon=10^{-6}$. It substitutes $1$ for the denominator's standard deviation term when $s<\epsilon$. The derivative there is $P/(1+\epsilon)$. A singleton follows the upstream reward divided by $1+\epsilon$. Alternative centered and RLOO derivatives are $P$ and $nP/(n-1)$ for non-singletons. The multi-turn Dr. Kernel GRPO branch uses the convention described in the related-work notes.

The chain rule gives $H=J\,\mathrm{diag}(f'(z))$. For log-speedup mean-error covariance $S$, the local objective is $\mathrm{tr}(H S H^T)$. A Gaussian working model predicts the change after observing candidate $i$ by conditioning on a noisy observation with variance $m_i S_{ii}$, where $m_i$ is the count already acquired. The resulting covariance is

$$
S'= S-\frac{S_{:i}S_{i:}}{(m_i+1)S_{ii}}.
$$

The score is the nonnegative predicted risk reduction divided by observed mean block cost. B4 uses the same covariance update and predicts reduction in reward error. B5 uses the reduction in the sum of posterior pair-order error probabilities on log-speedup. Fixed allocation cycles through the smallest counts. Each policy gets a shared initial budget reservation rule and cannot access reference scoring data.

The marginal variances use a Bartlett HAC working estimate with IID and positive floors. When exposures provide four or more common epochs, matched residual correlations are shrunk toward the diagonal and scaled by the fraction of overlap. Real serial candidate captures have no common epochs. The covariance update treats a fresh observation as conditionally independent, a simplifying assumption under drift and data-dependent allocation.

For thresholded or clipped rewards and for near ties where the noise is comparable to dispersion, the scheduler can use common-random-number Monte Carlo risk under its Gaussian working model. Conditional variance reduction at a fixed current point is a heuristic for value of information. It neither integrates over the future posterior mean nor provides adaptive coverage guarantees. Zero-variance branches, clipped plateaus and threshold discontinuities can produce very different first-order and finite-noise answers.
