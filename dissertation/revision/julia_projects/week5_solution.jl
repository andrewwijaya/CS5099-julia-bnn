### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 093a22cb-36a4-41d8-9c0e-46cf876192fe
begin
	using Distributions
	using ForwardDiff
	using Zygote
	using Random
	using Flux
	using LinearAlgebra
	using Plots
	using StatsPlots
	using Turing
	using MCMCChains
	using StatsFuns
	using LaTeXStrings
	plotly()
end

# ╔═╡ 51c76828-f55e-11ec-3e26-61ba6bc94c5a
md"""

# A primer on Bayesian inference (3)

**Abstract** *We have seen the benefits of Bayesian inference in the past sessions. They are in general more*
* *numerically stable,* 
* *natural to interpret and* 
* *lastly but not least overfitting proof.* 
*The downside, however, is the computational difficulty of the posterior distributions:*

$$p(w|\mathcal D) = \frac{p(w) p(\mathcal D|w)}{\int p(w) p(\mathcal D|w) dw}$$

*The denominator of the posterior, also known as **normalising constant**, involves a high dimensional integration, which in most cases cannot be computed exactly.*

*Last time, we have seen how the problem can be avoided by using sampling based approach. The idea is to draw samples from the **non-normalised posterior** directly:*

$$w^{(i)} \sim \tilde p(w,\mathcal D)\triangleq p(w)p(\mathcal D|w),$$

*Markov Chain Monte Carlo (MCMC) samplers can be applied to efficiently draw (non-independent yet complying) samples from the posterior. Note the normalising constant is not required in MCMC samplers. As a result, all inference questions becomes Monte Carlo estimation.*


*This time, we will see how the problem can be reduced to an optimisation problem. And the technique is called **variational inference** in machine learning community. In a nutshell, we aim to find an approximating posterior (with simplified structure) $q(w)$ that is as close to the true posterior $p(w|\mathcal D)$ as possible:*

$$q(w; \psi) \leftarrow \arg\min_{\psi} \textit{d}(q(w), p(w|\mathcal D)),$$

*A typical choice for the distance measure is KL divergence. As the problem has been reduced to an optimisation problem, the inference becomes more computationally efficient and scalable than MCMC based approaches.*

You should read the following book chapters and notes
- MLAPP Chapter 21.1, 21.2, 21.3.1, 21.5
- MLPR by Iain Murray's MLPR course at the University of Edinburgh: [w10b](https://mlpr.inf.ed.ac.uk/2021/notes/w10b_variational_kl.html), [w10c](https://mlpr.inf.ed.ac.uk/2021/notes/w10c_variational_details.html)
"""

# ╔═╡ f5d9b1cc-63fc-4bdf-9e76-184f8d85cb60
md"""

## Part 1. VI in general

Based on the abstract, by choosing the forward KL divergence as the distance metric to minimise the discrepancy between the variational approximating distribution $q(w)$ and true posterior $p(w|\mathcal D)$, i.e. 

$$d(q, p) = \text{KL}(q\| p),$$

1. how VI avoids finding the normalising constant explicitly ?
2. what is evidence lower bound (ELBO) ?

"""

# ╔═╡ 06250275-93d3-4004-b2c5-37bc093a6dcb
md"""
**Solution:**

First a few notations, 

$$p(w|\mathcal D) = \frac{1}{Z_0} p(w, \mathcal D),$$ 

where 

$Z_0\triangleq p(\mathcal D) =\int p(w, \mathcal D) dw$ is the normalising constant. 


To minimise the forward KL between $q(w)$ and $p(w|\mathcal D)$:

$$\begin{align}\text{KL}[q\|p] &= \int q(w) \ln \frac{q(w)}{p(w|\mathcal D)} dw\\
&= \int q(w) \ln q(w)dw - \int q(w) \ln p(w|D)dw\\
&= -\text H(q) - \int q(w) \ln \frac{p(w, \mathcal D)}{p(\mathcal D)} dw \\
&= -\text H(q) - \int q(w) \ln p(w, \mathcal D) dw + \int q(w) \ln p(\mathcal D) dw\\
&= -\text H(q) - \int q(w) \ln p(w, \mathcal D) dw + \ln p(\mathcal D),
\end{align}$$

where $H(\cdot)$ denotes the differential entropy of a distribution.

Therefore, optimising the KL divergence reduces to 

$$\begin{align}q \leftarrow \arg\min_{q} \text{KL}[q\|p] &= \arg\min_{q} \left\{ - \int q(w) \ln p(w, \mathcal D) dw-\text H(q)\right \}\\
&= \arg\min_{q} \left\{ -\langle \ln p(w,\mathcal D)\rangle_q-\text H(q)\right \}\\
&= \arg\max_{q} \left\{ \langle \ln p(w,\mathcal D)\rangle_q +\text H(q)\right \}
\end{align}$$


Therefore, we do not need to know the normalising constant to find the optimal variational distribution. The loss can be interpreted as the sum of negative (expected) log full joint term $- \langle \ln p(w, \mathcal D) \rangle_{q} =- \int q(w) \ln p(w, \mathcal D) dw$ and a regularisation penalty $-\text H(q)$.

That is we want, on average sense, to maximise the full joint term, but at the same time we do not want a overly confident variational distribution $q$ (note entropy measures uncertainty of a distribution, larger entropy means more random distribution). 

"""

# ╔═╡ ad647b20-9e7a-4b53-bbc5-7ff0c49c2a3c
md"""

**ELBO & Variational free energy**. Note that KL divergence is always positive, therefore 


$$\begin{align}\text{KL}[q\|p] 
&= -\text H(q) - \int q(w) \ln p(w, \mathcal D) dw + \ln p(\mathcal D) \geq 0,
\end{align}$$ which in turn implies

$$\ln p(\mathcal D) \geq \text H(q) + \int q(w) \ln p(w, \mathcal D) dw \triangleq \text{ELBO}$$

The model evidence is lowered bounded by the optimisation objective: $\text H(q) + \int q(w) \ln p(w, \mathcal D) dw.$ The function is therefore called ELBO. 

VI therefore can also be viewed as a procedure to approximate the log model evidence $\ln p(\mathcal D)$,  which is intractable in most general cases, *from below*.


The negative ELBO is called *variational free energy* in quantum physics community:

$$\mathcal F(q) \triangleq - \text{ELBO}(q) = - \text{H}(q(w)) - \int q(w) \ln p(w, \mathcal D) dw$$
"""

# ╔═╡ de714c36-1eac-4a92-8d6c-77947c245894
md"""
## Part 2. Mean field VI

Mean field VI assumes $q(w) = \prod_{i} q(w_i)$, i.e. the approximating variational distribution is fully independent. Then variational calculus tells us the best approximating distrbution can be found by setting each dimension of $q$, $q(w_i)$ by the expectation of the log full joint distribution: i.e.

for $i \in 1, \ldots, d$

$$q(w_i) = E_{\{w_{j, j\neq i}\}}[\ln p(\mathcal D, w)].$$ 


Consider the full Bayesian model for the fixed basis expansion model: i.e. given fixed basis expanded design matrix $\Phi$, and targets $\mathbf y$, the observations are of Gaussian form with noise level $\sigma^2 = 1/\beta$, 

$$p(\mathbf y| \Phi,  \theta,\beta) = N(\Phi  \theta, \beta^{-1}\mathbf I_{N})$$
and we further impose the following (hyper-)priors on the unknown parameters: 

$p(\theta) = N(\boldsymbol 0, \text{diag}(\boldsymbol \alpha)^{-1}) = \prod_{d=1}^D N(\theta_d; 0, \alpha_d^{-1})$

$$p(\boldsymbol \alpha) = \prod_d p(\alpha_d) = \prod_d \text{Gamma}(\alpha_d; a_0, b_0) = \prod_{d=1}^D \frac{b_0^{a_0}}{\Gamma(a_0)} \alpha_d^{a_0-1} e^{-b_0 \alpha_d}$$

$$p(\beta) = \text{Gamma}(c_0, d_0)= \frac{d_0^{c_0}}{\Gamma(c_0)} \beta^{c_0-1} e^{-d_0 \beta}$$

1. derive the MFVI algorithm for the problem.
2. what's the algorithm's connection to Gibbs sampling ?
3. why this approach cannot be directly applied to more general models, say logistic regression or neural networks ?

"""

# ╔═╡ 141fccea-1712-4a19-a0ad-e29e83ce7bc6
begin
	# generate the hidden signal 
	function signal_(x, k= -0.2)
		if (x> -1) && (x < 1) 
			return 1
		elseif (x>-8) && (x<-7)
			return k * x + (1.6- k*(-8))
		else
			return 0
		end
	end
	N = 50
	σ² = 4*1e-2
	# xsamples = rand(N) * 20 .- 10 
	xsamples = collect(range(-10., 10., length=N))
	# xsamples = shuffle(xsamples_)[1:N]
	# generate targets ts or y
	ts = signal_.(xsamples) + rand(Normal(0, sqrt(σ²)), N)
end;

# ╔═╡ cd2de8de-961a-4365-bac0-3b27b9041159
begin
	xs = -10:0.1:10
	# Φₜₑₛₜ = basis_expansion(xs, xsamples, σϕ², intercept)
	tₜₑₛₜ = signal_.(xs)
	# wₘₗ = (Φ'*Φ)^(-1)*Φ'*ts
	plot(xs, tₜₑₛₜ, ylim= [-0.8, 2.5], linecolor=:black, linestyle=:dash, label="Original")
	scatter!(xsamples, ts, markershape=:xcross, markersize=2.5, markerstrokewidth=0.1, markercolor=:gray, markeralpha=0.9, label="Obvs")

end

# ╔═╡ 51e61619-585e-4699-99f0-65ad18ec165b
md"""
**Solution**
1). The full joint is

$p(\theta, \boldsymbol \alpha, \beta, \mathbf y|\Phi) = p(\boldsymbol \alpha)p(\beta)p(\theta|\boldsymbol \alpha)p(\mathbf y|\Phi, \theta, \beta)$

Taking log, we have 

$$\ln p(\theta, \boldsymbol \alpha, \beta, \mathbf y|\Phi) = \sum_{d} \ln p(\alpha_d) +\ln p(\beta) +\ln p(\theta|\boldsymbol \alpha) +\ln p(\mathbf y|\Phi, \theta, \beta)$$


MFVI assumes an approximating VI distribution which are factorised or independent:

$$q(\theta, \boldsymbol\alpha, \beta) = q(\boldsymbol \alpha) q(\theta) q(\beta) = \prod_{d} q(\alpha_d) q(\theta) q(\beta),$$

and variational calculus tells us: to minimise the KL divergence, we only need to iteratively fix each variational distribution to its expectation.


Take $\theta$ for example, 

$$\begin{align}q^\ast(\theta) &\leftarrow E_{q(\beta, \boldsymbol \alpha)} [\ln p(\theta, \boldsymbol \alpha, \beta, \mathbf y|\Phi) ]\\
&= E_{q(\beta, \boldsymbol \alpha)} \left [\sum_{d} \ln p(\alpha_d) +\ln p(\beta) +\ln p(\theta|\boldsymbol \alpha) +\ln p(\mathbf y|\Phi, \theta, \beta)\right ] \\
&= \underbrace{\sum_d \langle\ln p(\alpha_d) \rangle_{q(\alpha_d)} + \langle\ln p(\beta) \rangle_{q(\beta)} }_{\text{const.}} + \langle \ln p(\theta|\boldsymbol \alpha)\rangle_{q(\boldsymbol \alpha)} + \langle\ln p(\mathbf y|\Phi, \theta, \beta) \rangle_{q(\beta)}  \\
&= \langle \ln p(\theta|\boldsymbol \alpha)\rangle_{q(\boldsymbol \alpha)} + \langle\ln p(\mathbf y|\Phi, \theta, \beta) \rangle_{q(\beta)}  + C\\
&= -\frac{1}{2}  \langle \theta^\top \text{diag}(\boldsymbol \alpha)\theta
\rangle_{q(\boldsymbol \alpha)} -\frac{1}{2} \left \langle\beta (\mathbf y-\Phi\theta)^\top (\mathbf y -\Phi\theta)\right \rangle_{q(\beta)}+ C \\
&= -\frac{1}{2}  \theta^\top \langle \text{diag}(\boldsymbol \alpha)\rangle_{q(\boldsymbol \alpha)}\theta
 -\frac{\langle \beta \rangle}{2} (\mathbf y-\Phi\theta)^\top (\mathbf y -\Phi\theta) + C, \\
\end{align}$$

By completing the squares, we can find the posterior is a Gaussian:


$$q^{\ast}(\theta) = N(\mathbf m_N, \mathbf C_N),$$

where 


$\mathbf m_N = \mathbf C_N (\langle \beta \rangle\Phi^\top \mathbf y), \mathbf C_N = (\text{diag}(\langle \boldsymbol \alpha\rangle) + \langle \beta\rangle\Phi^\top \Phi)^{-1}$

Check [week 2's](https://lf28.github.io/MSc_2022/Bayes1_sol.jl.html) solution for more details. Note that you can simply set $\mathbf m_0 = \mathbf 0, \mathbf C_0^{-1} = \text{diag}(\langle \boldsymbol \alpha \rangle),$ and $\Sigma^{-1} = \langle \beta\rangle I_N$ to recover the above result.
"""

# ╔═╡ 0ccdd961-9eb4-4854-90d6-7b41dda103a4
md"""

**Further details.** Similarly, we can find the VI for $\boldsymbol \alpha, \beta$, due to the conjugate priors being used, they are all of Gamma form. And you should verify it. 

To make it complete, their VI are

$$q(\beta) = \text{Gamma}(c_N, d_N),$$ where 

$$c_N = c_0 + \frac{N}{2}, d_N = d_0+\frac{1}{2} \langle ( \mathbf y - \Phi\theta)^\top ( \mathbf y - \Phi\theta)\rangle_{q(\theta)},$$


The required expectation can be computed as

$$\begin{align}\langle ( \mathbf y - \Phi\theta)^\top ( \mathbf y - \Phi\theta)\rangle_{q(\theta)} &= \langle \mathbf y^\top \mathbf y -2 \mathbf y^\top  \Phi\theta+ \theta^\top\Phi^\top \Phi\theta \rangle_{q(\theta)}\\
&= \mathbf y^\top \mathbf y -2 \mathbf y^\top  \Phi \langle \theta\rangle_{q(\theta)} + \langle \text{tr}(\theta^\top\Phi^\top \Phi\theta)\rangle_{q(\theta)} \\
&= \mathbf y^\top \mathbf y -2 \mathbf y^\top  \Phi \langle \theta\rangle_{q(\theta)} + \langle \text{tr}(\Phi^\top \Phi\theta\theta^\top)\rangle_{q(\theta)} \\
&= \mathbf y^\top \mathbf y -2 \mathbf y^\top  \Phi \langle \theta\rangle_{q(\theta)} +  \text{tr}\left (\Phi^\top \Phi \langle\theta\theta^\top\rangle_{q(\theta)} \right )
\end{align}$$

The expected value of a Gaussian is just its mean $\langle \theta \rangle_q = \mathbf m_N,$ and the second moment can be calculated via $\mathbf C_N = \langle \theta\theta^\top\rangle - \langle \theta\rangle  \langle \theta\rangle^\top$ therefore $\langle \theta\theta^\top\rangle_q= \mathbf C_N + \langle \theta \rangle_q  \langle \theta \rangle_q ^\top.$

Therefore, the expectation can also written in a neater form as:

$$\begin{align}\langle ( \mathbf y - \Phi\theta)^\top ( \mathbf y - \Phi\theta)\rangle_{q(\theta)} 
&= \mathbf y^\top \mathbf y -2 \mathbf y^\top  \Phi \langle \theta\rangle_{q(\theta)} +  \text{tr}\left (\Phi^\top \Phi \langle\theta\theta^\top\rangle_{q(\theta)} \right )\\
&= \mathbf y^\top \mathbf y -2 \mathbf y^\top  \Phi \langle \theta\rangle_{q(\theta)} +  \text{tr}\left (\Phi^\top \Phi \left (\mathbf C_N + \langle \theta \rangle_q  \langle \theta \rangle_q ^\top\right ) \right )\\
&= \mathbf y^\top \mathbf y -2 \mathbf y^\top  \Phi \langle \theta\rangle_{q(\theta)} +\text{tr}(\Phi^\top\Phi \langle \theta \rangle_q  \langle \theta \rangle_q ^\top) + \text{tr}(\Phi^\top\Phi \mathbf C_N) \\
&= \mathbf y^\top \mathbf y -2 \mathbf y^\top  \Phi \langle \theta\rangle_{q(\theta)} +\langle \theta \rangle_q ^\top \Phi^\top\Phi \langle \theta \rangle_q   + \text{tr}(\Phi^\top\Phi \mathbf C_N) \\
&= (\mathbf y - \Phi \mathbf m_N )^\top ( \mathbf y - \Phi \mathbf m_N ) + \text{tr}(\Phi^\top\Phi \mathbf C_N).
\end{align}$$
"""

# ╔═╡ ebcca522-2f3f-4d89-bd7a-96f72dba8056
md"""

and $$q(\alpha_d) = \text{Gamma}(a_d, b_d)$$ where

$$a_d = a_0 + \frac{1}{2}, b_d = b_0 +\frac{1}{2} \langle \theta_d^2\rangle_{q(\theta)}.$$

Gamma distribution's expectations are widely known:

$\langle \beta \rangle = \frac{c_N}{d_N}.$

Similarly, for $\alpha_d$, its expectation under a Gamma variational distribution $q(\alpha_d) = \text{Gamma}(a_d, b_d)$ is 

$$\langle \alpha_d \rangle = \frac{a_d}{b_d}.$$
"""

# ╔═╡ 41e52510-2d17-4a4d-b6f5-c452d7f84883
md"""

The MFVI algorithm for this particular model is

*repeat until converge* 

  1. *update variational distribution for $\theta$:*

$$q^\ast(\theta)\leftarrow \mathcal N(\mathbf m_N, \mathbf C_N)$$

  2. *update variational distribution for $\boldsymbol \alpha$:*

$$q^\ast(\boldsymbol \alpha) \leftarrow \prod_{d} \text{Gamma}(a_d, b_d)$$

  3. *update variational distribution for $\beta$:*

$$q^\ast(\beta) \leftarrow \text{Gamma}(c_N, d_N)$$
"""

# ╔═╡ d29c0514-6a1e-4516-9d27-a0674483e5b5
md"""

2). Relation to Gibbs sampling,


Gibbs sampling aims at *sampling* from full conditionals,

$$x \sim p(x|\text{mb}(x)),$$


MFVI aims at *fixating* the full conditionals to its expectation,

$$q(x) \leftarrow \exp\{E_{q(\text{mb}(x))}[\ln  p(x|\text{mb}(x))]\}.$$

"""

# ╔═╡ 747f77f9-eda4-4a49-8428-88c1b91a68c9
md"""

As a result, the algorithm of Gibbs sampling is very similar to the MFVI:

*repeat until converge* 

  1. *sample the full conditional distribution for $\theta$:*

$$\theta^{(m)}\sim \mathcal N(\theta; \mathbf m_N, \mathbf C_N)$$

  2. *sample the full conditional distribution for $\boldsymbol \alpha$:*

$$\boldsymbol \alpha^{(m)} \sim \prod_{d} \text{Gamma}(\alpha_d; a_d, b_d)$$

  3. *sample the full conditional distribution for $\beta$:*

$$\beta^{(m)} \sim \text{Gamma}(\beta; c_N, d_N)$$


Also note that the distribution parameters are different for the two algorithms. Gibbs sampling rely on the samples from other conditionals while MFVI uses the expectations from the other mean field variational distributions.
"""

# ╔═╡ f30b5b09-860e-4dc7-91c4-2dff8c6608a3
begin
	# Radial basis function
	# called Gaussian kernel in the paper
	function rbf_basis_(x, μ, σ²)
		exp.(-1 ./(2*σ²) .* (x .- μ).^2)
	end

	# vectorised version of basis expansion by RBF
	function basis_expansion(xtest, xtrain, σ²=1, intercept = false)
		# number of rows of xtrain, which is the size of basis
		n_basis = size(xtrain)[1] 
		n_obs = size(xtest)[1]
		X = rbf_basis_(xtest, xtrain', σ²)
		intercept ? [ones(n_obs) X] : X
	end

end

# ╔═╡ fcb61290-3de5-403a-9183-7668baf6dec6
begin
	intercept = true
	σϕ²= 0.5
	Φ = basis_expansion(xsamples, xsamples, σϕ², intercept);
	print()
end

# ╔═╡ 7813bb32-8b29-442d-814b-eb513e50d526
function viBayesianLR(Φ, ts, a0=0, b0=0, c0=1e-6, d0=1e-6, maxIters=100)
	N,M = size(Φ)
	innerM = Φ'*Φ
	# initialise starting points
	αt = 10 .* ones(M)
	βt = var(ts)
	# initialisation for q(β)
	ct = c0 + N/2
	dt = ct * var(ts) 
	# initialisation for q(α)
	at = a0 + 0.5
	bt = b0 .+ 0.5 * ones(M)
	μt = zeros(M)
	Σt = Diagonal(ones(M))
	for i in 1:maxIters
		# update q(w)
		β̂ = ct/dt 
		Â = Diagonal(at ./bt)
		invΣt = β̂ * innerM + Â
		Σt = inv(invΣt)
		μt = β̂ * Σt * Φ' * ts
		MM = Σt + μt * μt'
		# update q(β)
		# dt = d + 0.5 * sum(ts.^2) - sum(μt .* (Φ'*ts)) + 0.5 * sum((Φ * MM) .* Φ)
		dt = d0 + 0.5 * (sum((ts-Φ*μt).^2) + tr(innerM * Σt))
		# update q(α)
		bt = b0 .+ 0.5 * (μt.^2 + diag(Σt))
		# should check convergence here !
	end
	return μt, Σt, at, bt, ct, dt
end

# ╔═╡ f0b3f4d9-2a54-4351-bb73-30320b3fc58e
begin
	a0 = 1e-6
	b0 = 1e-6
	c0 = 1e-6
	d0 = 1e-6
	wᵥᵢ, Σᵥᵢ, aᵥᵢ, bᵥᵢ, cᵥᵢ, dᵥᵢ=viBayesianLR(Φ, ts, a0, b0, c0, d0);
end;

# ╔═╡ b36d4f31-95ea-4ce6-9409-e3ee1da9c24e
plot(wᵥᵢ, label="μ")

# ╔═╡ f62b5fee-4765-466e-9f26-02a5b119a1e8
md"""

3).
MFVI would fail as the approximating distribution is of unknown form, and the expectation has no closed form solution.

Take logistic regression for example, the likelihood is not exponential quadratic but logistic transformation of a linear function of $\theta$. The posterior therefore is no longer Gaussian: i.e. we cannot massage the following term into a quadratic function:

$$q^\ast(\theta) = \langle \ln p(\theta|\boldsymbol \alpha)\rangle + \langle \ln p(\mathbf y|\Phi, \theta)\rangle$$


For neural networks, the likelihood term can be even more complicated due to the non-linear transformations used: $f(\phi_i)$, the likelihood will never by quadratic. 

Then the whole iterative update procedure falls apart, e.g. $$\langle \theta \rangle$$ is unknown. 
"""

# ╔═╡ 8ab14983-73fd-4b85-82ad-a0b3404f5918
md"""

## Part 3. Fixed form VI or pathwise gradient

Read Iain Murray's notes on VI. He has also provided a Python implementation for a general fixed form VI inference problem. Inspect and understand the code. Port the implementation to Julia.


1. use fixed form VI to solve the above fixed basis expansion regression problem. 
2. solve Bayesian logistic regression problem by fixed form VI. 
"""

# ╔═╡ 0ed401f5-097c-4206-8336-b751c3b8da17
begin
	D1 = [
	    7 4;
	    5 6;
	    8 6;
	    9.5 5;
	    9 7
	]
	
	D2 = [
	    2 3;
	    3 2;
	    3 6;
	    5.5 4.5;
	    5 3;
	]

	D = [D1; D2]
	D = [ones(10) D]
	targets = [ones(5); zeros(5)]
	plot(D1[:,1], D1[:,2], xlabel=L"$x_1$", ylabel=L"x_2", label="class 1", seriestype=:scatter, ratio=1.0)
	plot!(D2[:,1], D2[:,2], xlabel=L"$x_1$", ylabel=L"x_2", label="class 2", seriestype=:scatter, ratio=1.0)
	# scatter!(D2[:,1], D2[:,2], label="class 0")
end

# ╔═╡ d7646301-f225-4319-839b-855140801f54
md"""

**Solution**

The variational free energy is 



$$\mathcal F(q) = -\text H(q) - \langle \ln p(\mathbf w, \mathcal D) \rangle_q,$$

If we fix the approximating variational distribution to be a Gaussian, i.e. $q(\mathbf w) = N(\mathbf m, \mathbf V),$ we can optimise the variational free energy w.r.t the variational distribution $q$ via its parameters $\mathbf{m, V}$ instead. $\mathbf{m},\mathbf{V}$ are called variational parameters.

To optimise a symmetric positive matrix is hard, instead we reparameterise the variance by cholesky decomposition:

$\mathbf V = \mathbf L\mathbf  L^\top,$ where $\mathbf L$ is a lower triangular matrix:

$$\mathbf L = \begin{bmatrix}L_{11}& 0 &0 & \ldots& 0\\
L_{21}& L_{22}& 0&\ldots& 0 \\
\vdots& \vdots& \vdots&\vdots& \vdots \\
L_{D1}& L_{D2}& L_{D3}&\ldots& L_{DD}\end{bmatrix}$$

To make the computation more stable, we constrain the diagonal entries of $\mathbf L$ to be strictly positive, and reparameterise them by $\exp, \ln$ transformation:

$$\mathbf L^\ast =\begin{bmatrix}\exp(L_{11}^\ast)& 0 &0 & \ldots& 0\\
L_{21}^\ast& \exp(L_{22}^\ast)& 0&\ldots& 0 \\
\vdots& \vdots& \vdots&\vdots& \vdots \\
L_{D1}^\ast& L_{D2}^\ast& L_{D3}^\ast&\ldots& \exp(L_{DD}^\ast)\end{bmatrix}$$

Therefore, the variational parameters become $\mathbf m, \mathbf L^\ast,$ where $L_{dd} = \exp(L^\ast_{dd}),$ and $L_{dd'} = L_{dd'}^\ast$ for $d\neq d'$.

You can think the transformation as a multivariate version of reparameterisation of $\sigma^2$:  $\sigma^2 = \exp(\ln \sigma) \cdot \exp(\ln\sigma)).$ And we optimise $\ln \sigma$ instead. 
"""

# ╔═╡ 084f17cb-8301-4708-9c81-c2b8c5f041f7
md"""

Since we have assumed $q$ is Gaussian, we can find its entropy in closed form:


$$\text H(q) = \frac{D}{2} +\frac{1}{2} \ln |2\pi \mathbf V| = \frac{D}{2} +\frac{D}{2} \ln(2\pi)  + \sum_{d} \ln L_{dd}$$

The free energy becomes 

$$\mathcal F(q) = - \sum_d \ln L_{dd} - \langle \ln p(\mathbf w, \mathcal D)\rangle_q + C.$$

*Note that I have not assumed the prior is Gaussian here. Therefore, the model here is more general. We can for example assume $p(\mathbf w)$ is Laplace distributed instead of Gaussian to achieve more sparse posterior (Laplace prior corresponds to Bayesian LASSO).* 

Next, we take gradient of $\mathcal F$ w.r.t $\mathbf{m}, \mathbf L:$

$$\nabla_{\mathbf m} \mathcal F = - \nabla_{\mathbf m}  \langle \ln p(\mathbf w, \mathcal D)\rangle_q, $$


$$\nabla_{\mathbf L} \mathcal F = - \nabla_{\mathbf L} \left (\sum_d \ln L_{dd}\right )- \nabla_{\mathbf L}  \langle \ln p(\mathbf w, \mathcal D)\rangle_q,$$

The gradient of the negative entropy term is:

$\frac{\partial \sum_d \ln L_{dd}}{\partial L_{ij}} =  L_{ij}^{-1}\cdot \delta_{i=j},$

In matrix notation:

$\nabla_{\mathbf L} \left ( \sum_d \ln L_{dd} \right ) = \text{diag}(\{L_{dd}^{-1}\})\circ \mathbf I_D$

"""

# ╔═╡ c915353c-3cdc-4949-93a9-3b8b21e3ba4f
md"""
We use the reparameterisation trick or path-wise gradient to side-step the difficulty of calculating the expectation.

First note, if $\nu \sim N(\mathbf 0, \mathbf I),$ then $\mathbf w \triangleq \mathbf m + \mathbf L \nu \sim N(\mathbf m, \mathbf V)$, therefore

$$\langle f(\mathbf w)\rangle_{\mathbf w\sim \mathcal N(\mathbf m, \mathbf V)} = \langle f(\mathbf m + \mathbf L \nu)\rangle_{\nu \sim \mathcal N(\mathbf 0, \mathbf I)}$$


We can therefore propogate the gradient through the expectation:

$$\begin{align}\nabla_{\mathbf m} \langle f(\mathbf w)\rangle_{\mathbf w\sim \mathcal N(\mathbf m, \mathbf V)} &= \nabla_{\mathbf m}\langle f(\mathbf m + \mathbf L \nu)\rangle_{\nu \sim \mathcal N(\mathbf 0, \mathbf I)}= \langle \nabla_{\mathbf m}f(\mathbf m + \mathbf L \nu)\rangle_{\nu \sim \mathcal N(\mathbf 0, \mathbf I)} \\
&=\langle \nabla_{\mathbf m + \mathbf L \nu}f(\mathbf m + \mathbf L \nu)\rangle_{\nu \sim \mathcal N(\mathbf 0, \mathbf I)}
\end{align}$$

We have exchanged the gradient and expectation order in the second last step, which holds under regularity conditions.

For $\mathbf L$, we have 


$$\begin{align}\nabla_{\mathbf L} \langle f(\mathbf w)\rangle_{\mathbf w\sim \mathcal N(\mathbf m, \mathbf V)} &= \langle \nabla_{\mathbf L}f(\mathbf m + \mathbf L \nu)\rangle_{\nu \sim \mathcal N(\mathbf 0, \mathbf I)} \\
&=\langle \nabla_{\mathbf m + \mathbf L \nu}f(\mathbf m + \mathbf L \nu) \nu^\top \circ \mathbf S_{L}\rangle_{\nu \sim \mathcal N(\mathbf 0, \mathbf I)}
\end{align}$$

where $$\mathbf S_{\mathbf L} = \begin{bmatrix}1& 0 &0 & \ldots& 0\\
1& 1& 0&\ldots& 0 \\
\vdots& \vdots& \vdots&\vdots& \vdots \\
1& 1& 1&\ldots& 1\end{bmatrix}.$$ The hadamard product just set the +1 upper triangular parts to be zero due the lower triangular structure of the matrix $\mathbf L$.

"""

# ╔═╡ cfdeee66-e4c4-48fc-af95-287e70fb0ec8
md"""

**Unbiased Monte Carlo gradient approximation.** To approximate the gradient, we first sample 

$\nu^{(m)} \sim \mathcal N(\mathbf 0, \mathbf I_D),$ then propogate to 

$\mathbf w^{(m)} = \mathbf m + \mathbf L \nu^{(m)}.$ The pathwise Monte Carlo gradient is

$$\nabla_{\mathbf m} \mathcal F = - \nabla_{\mathbf m}  \langle \ln p(\mathbf w, \mathcal D)\rangle_q = - \langle \nabla_{\mathbf w}   \ln p(\mathbf w, \mathcal D)\rangle_{\nu\sim N(\mathbf 0, \mathbf I)} \approx -\frac{1}{M} \sum_{m} \nabla_{\mathbf w^{(m)}}   \ln p(\mathbf w^{(m)}, \mathcal D)$$

$$\begin{align}\nabla_{\mathbf L} \mathcal F &= - \text{diag}(\{L_{dd}^{-1}\})\circ \mathbf I_D- \nabla_{\mathbf L}  \langle \ln p(\mathbf w, \mathcal D)\rangle_q\\
&= -\text{diag}(\{L_{dd}^{-1}\})\circ \mathbf I_D- \langle \nabla_{\mathbf w}   \ln p(\mathbf w, \mathcal D) \nu^\top\rangle_{\nu \sim \mathcal N(\mathbf 0, \mathbf I_D)}\\
&= -\text{diag}(\{L_{dd}^{-1}\})\circ \mathbf I_D- \frac{1}{M} \sum_{m} \left \{\left [\nabla_{\mathbf w^{(m)}}   \ln p(\mathbf w^{(m)}, \mathcal D)\right ] (\nu^{(m)})^\top \circ \mathbf S_{\mathbf L}\right \} 
\end{align}$$
"""

# ╔═╡ 8b009cd4-bafc-48aa-9dcd-df92e13d4b6d
md"""

Finally, note that due to reparameterisation of the diagonal entries of $\mathbf L,$ the diagonal entries's gradient need to be multiplied by $\exp(L^\ast_{dd})$, where we have just simply applied chain rule.

$$\begin{align}\nabla_{\mathbf L^\ast} \mathcal F &= \nabla_{\mathbf L} \mathcal F
 \circ \text{diag}(\{\exp(L_{dd}^\ast)\})= \nabla_{\mathbf L} \mathcal F
 \circ \text{diag}(\{L_{dd}\})\end{align}$$

"""

# ╔═╡ 3f80ce90-c4d2-4085-80ae-f1ce1b0088a4
begin
	# Cholesky lower triangle and exp reparametrisation.
	function pack_L(LL)
		diagidx = diagind(LL)
		L = tril(LL)
		L[diagidx] = exp.(diag(LL))
		return L
	end
end

# ╔═╡ a1bb4425-17cf-4d62-836f-559020721217
# Stochastic Variational Inference. mm is the mean of the Gaussian posterior, LL is the reparametrised variance of the Gaussian posterior, l is the log (approximation) posterior function, mc is monte carlo samples, mf is mean field. If mf=true the assumption is that the variational distributions are independent. Fixed form means the assumption is the posterior is a Gaussian.
function svi_grad(mm, LL, ℓ; mc = 1, mf=false)
	D = size(mm)[1]
	diagidx = diagind(LL)
	if !mf
		# L = tril(LL)
		# L[diagidx] = exp.(diag(LL))
		L = pack_L(LL)
	else
		# If mean field do this instead.
		L = Diagonal(exp.(diag(LL)))
	end
	ℱ = -0.5 * D - 0.5 * D * log(2 * π) - sum(diag(LL))
	ν = randn(D, mc)
	ww = mm .+ L * ν
	#the l() function return log prob and gradient.
	ℓs, ∇ws = ℓ(ww)
	ℱ -= mean(ℓs)
	∇m = - mean(∇ws, dims=2)[:]
	∇L = - tril((1 / mc) .* ∇ws * ν') 
	∇L[diagidx] = (∇L[diagidx] .* L[diagidx]) 
	∇L -= I
	#  fixed form plus mean field assumption: i.e. isotropic diagonal Gaussian
	# there are faster ways to do MF optimisation!
	if mf
		∇L = Diagonal(diag(∇L)) 
	end
	return ∇m, ∇L, ℱ
end

# ╔═╡ 49f51c64-ffec-47f1-a332-09988442d5ed
begin
	#Scrap code, delete whenever
	A = [1 2 3;4 5 6;7 8 9]
	diagindx = diagind(A)
 	diag(A)
	A[diagindx] .+= 1
	diagindx = diagind(A)
end

# ╔═╡ 59154ad8-197c-4ea2-a0a4-fd0e3c862af1
md"""

**A simple example**

The above algorithm can be used to approximate any **unconstrained** distribution with a Gaussian. 
"""

# ╔═╡ fb9b8223-14ec-4e4c-b9a9-f9c44e912008
begin
	# unnormalised log target density to approximate
	function log_density(x)
		mu = x[1]
		log_sigma = x[2]
		sigma_density = logpdf(Normal(0, 1.35), log_sigma)
		mu_density = logpdf(Normal(0, exp(log_sigma)), mu)
		return sigma_density + mu_density
	end

	# find the gradient
	gx = x -> ForwardDiff.gradient(log_density, x); 
	# this is slow: ideally you should define a vectorised version 
	log_den_fun = (x) -> (mapslices(log_density, x, dims=1)[:], mapslices(gx, x, dims=1))
end

# ╔═╡ 8c2c59b1-063c-4010-af76-de63dec44717
begin
	dim =2
	# random initialisation of the variational parameters m and L.
	# This cell is not doing anything, just demonstrating the output of svi_grad.
	init_m = -1 * ones(dim)
	init_L = LowerTriangular(rand(dim, dim))
	svi_grad(init_m, init_L, log_den_fun; mc = 1, mf=true)
end

# ╔═╡ c7687b5f-0740-4dd9-b26b-8d29537ddced
begin
	# Initialise m and L with random guesses.
	m₀ = [-1, -1.]
	L₀ = [-5 0.01; 0.01 -5]
	opt = ADAM(0.01, (0.9, 0.999))
	iters = 1000
	mf = true
	# Initialise the free energy/elbo list with zeros.
	ℱs = zeros(iters) 
	for t in 1:iters
		# For each epoch, get gradient of m and L and populate the free energy list.
		∇m, ∇L, ℱs[t] = svi_grad(m₀, L₀, log_den_fun; mc = 50, mf=mf)
		Flux.Optimise.update!(opt, m₀, ∇m)
		Flux.Optimise.update!(opt, L₀, ∇L)
	end
end

# ╔═╡ 2d85dce1-d532-42f9-9f1c-33932f3f425f
m₀ # It has optimised the mean to this 2d vector point.

# ╔═╡ caefb087-9004-4762-8026-c9717ee1187e
begin
	L = pack_L(L₀)
end

# ╔═╡ ef971f6b-844a-44ca-82ec-61cf00f80114
L₀

# ╔═╡ 75f28936-2b95-4103-a06b-fd4c4908919a
begin
	x₁ = range(-2, stop=2, length=151)
	x₂ = range(-4, stop=2, length=151)
	plot(x₁,x₂, (x, y) -> exp(log_density([x, y])), st=:contour)
	if !mf
		plot!(x₁,x₂, (x, y) -> pdf(MvNormal(m₀, L *L'), [x,y]), st=:contour)
	else
		plot!(x₁,x₂, (x, y) -> pdf(MvNormal(m₀, Diagonal(diag(L)) *Diagonal(diag(L))'), [x,y]), st=:contour)
	end
end

# ╔═╡ 7446506a-17fa-4c27-9ef4-dcef0057202a
plot(-ℱs, label="ELBO")

# ╔═╡ 9ceb3db3-da5e-47c8-9244-89c213d8f0b4
begin
	# Similar to cell above, but modularised. This is the top level ffvi function that takes some lfun variational posterior to optimise.
	function ffvi(ℓfun, dim; mc=1, maxiters=5000, meanfield = false)
		m = randn(dim)
		L = randn(dim, dim)
		opt = AMSGrad(0.01, (0.9, 0.999))
		ℱs = zeros(maxiters) 
		for t in 1:maxiters
			∇m, ∇L, ℱs[t] = svi_grad(m, L, ℓfun; mc = mc, mf=meanfield)
			Flux.Optimise.update!(opt, m, ∇m)
			Flux.Optimise.update!(opt, L, ∇L)
		end
		return m, L, ℱs
	end
end

# ╔═╡ a3a01e98-c8fc-428d-9264-9759fabe6826
begin
	# Believe this is the posterior = prior * likelihood
	function ℓ_fixed_basis(w, Φ, ts, m₀, α, β)
		#Gradient of prior + gradient of likelihood
		∇ℓ = - α .* w + β .* (Φ' * (ts .- Φ * w))
		#println("ts", ts)
		#println("Φ", Φ)
		#println("w", w)
		logpdf(MvNormal(m₀, sqrt.(1.0 ./ α)), w)  + sum(-0.5 .* β .* (ts .- Φ*w).^2, dims=1)[:], ∇ℓ
	end

	# Ignore for now.
	# Laplace prior with fixed basis expansion model
	function ℓ_fixed_basis_2(w, Φ, ts, α, β)
		dim = length(w)
		# logpdf(filldist(LocationScale(0, sqrt(1/α), TDist(2)), dim), w)+ sum(-0.5 * β * (ts - Φ*w).^2)
		logpdf(filldist(LocationScale(0, sqrt(1/α), Laplace()), dim), w)+ sum(-0.5 * β * (ts - Φ*w).^2)
		# logpdf(filldist(LocationScale(0, sqrt(1/α), Laplace()), dim), w)+ sum(-0.5 * β * abs.(ts - Φ*w))
	end

end

# ╔═╡ ae10fc12-8ddc-4eab-815d-64fd49f5b37d
begin
	#dim fb is 51 - includes bias term? First element is obs (50)?
	dim_fb = size(Φ)[2]
	# alpha beta hardcoded to 1
	ℓ_fb(w) = ℓ_fixed_basis(w, Φ, ts, zeros(dim_fb) ,1.0, 1.0)
	m_fb , L_fb , ℱs_fb = ffvi(ℓ_fb, dim_fb; mc = 1, meanfield=false)
end

# ╔═╡ 9390258d-37d0-4b42-b80b-676ff6b384df
L_fb

# ╔═╡ 40283cc9-8b14-42c8-8b5c-e3d8114cfd12
m_fb

# ╔═╡ 7bd6e1f7-dbc9-4824-b67b-016d6121bcde
Φ

# ╔═╡ 03c9a889-108a-4f4e-90e2-29526c3c6ead
plot(-ℱs_fb)

# ╔═╡ caafef71-22db-4631-98af-9d4eb39e5acd
md"""

**Hyperparameter update**

We can also optimise the hyperparameters together with the variational parameter. For Gaussian prior models, the hyperparameters can be updated in closed form. We use the fixed basis expansion model below as an example.


Remember the hyperparameters are $$\sigma^2$$ and $\boldsymbol \lambda^2$. Take the gradient of $\mathcal F$ w.r.t $\sigma^2$ and $\boldsymbol\lambda^2$ and set them to zero: their closed form estimators are

$$\sigma^2 = \frac{1}{N} \langle (\mathbf y - \Phi \mathbf w)^\top(\mathbf y - \Phi \mathbf w)\rangle_q,$$

$$\lambda^2_{d} = \boldsymbol \Sigma_{dd} + m_d^2.$$

The derivation is left as an exercise.
"""

# ╔═╡ 94b16351-7968-486c-93e9-d80910741cfc
begin


	function ffvi_fixed_basis_regression(ℓlik, ℓprior, dim; mc=1, maxiters=5000, meanfield = false)
		m = zeros(dim)
		L = zeros(dim, dim)
		opt = AMSGrad(0.01, (0.9, 0.999))
		ℱs = zeros(maxiters) 
		λ² = 100.0 * ones(dim)
		σ² = 100.0
		for t in 1:maxiters
			function ℓfun(w) 
				logL, ∇l = ℓlik(w, σ²) 
				logP, ∇p = ℓprior(w, λ²)
				return logL + logP, ∇l+∇p
			end
			∇m, ∇L, ℱs[t] = svi_grad(m, L, ℓfun; mc = mc, mf=meanfield)
			Flux.Optimise.update!(opt, m, ∇m)
			Flux.Optimise.update!(opt, L, ∇L)
			# alternatively, gradient descent on hyperparameters after reparameterisation
			# Flux.Optimise.update!(opt, θ, ∇θ)
			# θ[θ .> 100] .= 100.0
			# update hyperparameter
			L0 = pack_L(L)
			Σ = L0 * L0'
			λ² = diag(Σ) + m.^2
			σ² = (sum((ts .- Φ * m).^2) + tr(Φ'*Φ * Σ))/ size(Φ)[1]
		end
		return m, L, ℱs, λ², σ²
	end

	function ℓ_fixed_basis_lik(w, Φ, ts, m₀, β)
		∇ℓ = β .* (Φ' * (ts .- Φ * w))
		sum(-0.5 .* β .* (ts .- Φ*w).^2, dims=1)[:], ∇ℓ
	end

	function ℓ_gaussian_prior(w, σ²; dim=nothing)
		if isnothing(dim)
			dim = length(σ²)
		end
		logpdf(MvNormal(zeros(dim), sqrt.(σ²)), w) , - (1 ./ σ²) .* w 
	end

end

# ╔═╡ 34a2a48d-8802-4fb8-bcac-41b3a4de793c
begin
	ℓ_lik(w, σ²) = ℓ_fixed_basis_lik(w, Φ, ts, zeros(dim_fb), 1/σ²)
	ℓ_prior(w, σ²) = ℓ_gaussian_prior(w, σ²; dim = dim_fb)
	m_fb_ , L_fb_ , ℱs_fb_, σ²_, s_ = ffvi_fixed_basis_regression(ℓ_lik, ℓ_prior, dim_fb;mc =100)

end

# ╔═╡ 7f229bb8-cfc5-4cc1-a603-2244dddad00d
plot(log.(1 ./ σ²_), label="log(α)")

# ╔═╡ ef8a496e-791c-47b5-84ee-af95c6d71df6
plot(-ℱs_fb_, label="ELBO")

# ╔═╡ 1b81b6cd-87d1-4c49-bd54-06c57c62182b
begin
	# Laplace prior model
	ℓ_fb_2(w) = ℓ_fixed_basis_2(w, Φ, ts, 2.0, 1.0)
	gx_fb2 = x -> ForwardDiff.gradient(ℓ_fb_2, x); 
	ℓ_fb_fun2 = (x) -> (mapslices(ℓ_fb_2, x, dims=1)[:], mapslices(gx_fb2, x, dims=1))
	m_fb2 , L_fb2 , ℱs_fb2 = ffvi(ℓ_fb_fun2, dim_fb)
end

# ╔═╡ d79dc084-0cd8-484e-bd17-3d0abce254e1
begin
	# ℓ_normal_ = (x) -> (mapslices(ℓ_normal, x, dims=1)[:], mapslices(g_full, x, dims=1))
	# m_normal , L_normal , ℱs_normal = ffvi(ℓ_normal_, 2)
	# m_normal = zeros(2)
	# L_normal = zeros(2, 2)
	# opt_normal = AMSGrad(0.01, (0.9, 0.999))
	# iters_normal = 2000
	# mf_normal = false
	# ℱs_normal = zeros(iters_normal) 
	# for t in 1:iters_normal
	# 	∇m, ∇L, ℱs_normal[t] = svi_grad(m_normal, L_normal, ℓ_normal_; mc = 10, mf=mf_normal)
	# 	Flux.Optimise.update!(opt, m_normal, ∇m)
	# 	Flux.Optimise.update!(opt, L_normal, ∇L)
	# end
	# plot(ℱs_normal)
end

# ╔═╡ ad5fc73a-e2b2-4784-ad74-94a2fb43befe
m_fb

# ╔═╡ 75b2f8a2-228a-4da4-92f7-a4890f53fc54
md"""

**3) Bayesian logistic regression by fixed form VI**
"""

# ╔═╡ a474cfda-755e-4fed-a6d3-441195c3ed3a
# vectorised version of log posterior of logistic regression
function logPosteriorLogisticR(w, m0, V0, X, y)
	σ = logistic.(X * w)
	Λ0 = inv(V0)
	grad = - Λ0  * (w .- m0) + X' * (y .- σ)
	return logpdf(MvNormal(m0, V0), w) .+ sum(logpdf.(Bernoulli.(σ), y), dims=1)[:], grad
end

# ╔═╡ 0e5ea9e0-c335-4309-92d9-bda23328a230
begin
	m0 = zeros(3)
	V0 = 100. .* Matrix(I, 3, 3)
end

# ╔═╡ d1b3c121-f35d-491c-834f-6c8de8410df0
m_lr , L_lr , ℱs_lr = ffvi((w) -> logPosteriorLogisticR(w, m0, V0, D, targets), 3;mc =10, meanfield=false)

# ╔═╡ e78b2b3c-2660-4651-b5d9-4d974af4d451
pack_L(L_lr)* pack_L(L_lr)'

# ╔═╡ 58cd8b62-5f55-4e7a-a5eb-ea372eb78e88
plot(-ℱs_lr)

# ╔═╡ 0a7e9d02-1a8b-4c9f-a61c-55c5276ecfef
md"""

## Appendix

"""

# ╔═╡ 15f1b0ab-448a-4bbf-9941-9502623681e5
begin
	# Gibbs sampling solution

	function samplew(t, Φ, m₀, C₀, β)
		C0Inv = inv(C₀)
		CnInv = C0Inv + β * (Φ'*Φ)
		Cn = inv(CnInv)
		mn = Cn*(β*Φ'*t + C0Inv * m₀)
		return rand(MvNormal(mn, Matrix(Symmetric(Cn))))
	end

		
	function sampleβ(sse, n, c0, d0)
		cn = c0 + n/2
		dn = d0 + sse/2
		return rand(Gamma(cn, 1/dn))
	end

	function sampleαs(w, a0, b0)
		an = a0 .+ 0.5
		bn = b0 + 0.5 * w.^2
		return rand.(Gamma.(an, 1 ./bn))
	end


	function gibbsBayesianLR(Φ, ts, a0, b0, c0, d0, mc=5000, burnin=2000)
		N,M = size(Φ)
		# place holders or storage
		ws = zeros(M, mc)
		αs = zeros(M, mc)
		βs = zeros(mc)
		# initialise starting points
		αt = 10 .* ones(M)
		βt = var(ts)
		# hyperparameters
		a0 = a0 .* ones(M)
		b0 = b0 .* ones(M)
		m0 = zeros(M)
		for i in 1:(mc+burnin)
			# sample weight w
			C0 = Diagonal( 1 ./ αt)
			wt = samplew(ts, Φ, m0, C0, βt)
			# sample β
			sse = sum((ts - Φ*wt).^2)
			βt = sampleβ(sse, N, c0, d0)
			# sample α
			αt = sampleαs(wt, a0, b0)

			if i > burnin
				ws[:,i-burnin] = wt
				αs[:,i-burnin] = αt
				βs[i-burnin] = βt
			end
		end
		return ws, αs, βs
	end

end

# ╔═╡ 60c96652-ffe5-444a-8cda-b5fae6472c77
ws, αs, βs = gibbsBayesianLR(Φ, ts, a0, b0, c0, d0);

# ╔═╡ c4351958-ef47-4c69-ba95-da52e7b54fb5
begin
	# xs = -10:0.1:10
	Φₜₑₛₜ = basis_expansion(xs, xsamples, σϕ², intercept)
	# tₜₑₛₜ = signal_.(xs)
	wₘₗ = (Φ'*Φ)^(-1)*Φ'*ts
	plot(xs, tₜₑₛₜ, ylim= [-0.8, 2.5], linecolor=:black, linestyle=:dash, label="Original")
	scatter!(xsamples, ts, markershape=:xcross, markersize=2.5, markerstrokewidth=0.1, markercolor=:gray, markeralpha=0.9, label="Obvs")
		
	plot!(xs, Φₜₑₛₜ*wₘₗ, linecolor=:green, linestyle=:solid, lw=1, label="ML")
		
	# plot!(xs, Φₜₑₛₜ*wₑₘ, linecolor=:red, linestyle=:solid, lw=2, label="EM")
	gibbsPredictions = Φₜₑₛₜ * ws
	plot!(xs, mean(gibbsPredictions, dims=2), linecolor=:blue, linestyle=:solid, lw=2, label="Gibbs")
	
	plot!(xs, Φₜₑₛₜ*wᵥᵢ, linecolor=:orange, linestyle=:solid, lw=2, label="VI")

	# gibbsFullPred = predict_basis(xs, wϕs, σϕs, xsamples, intercept)
	# plot!(xs, mean(gibbsFullPred, dims=2), linecolor=:cyan3, linestyle=:solid, lw=2, label="Full Bayes")

	# xtest_ = basis_expansion(xs, xsamples, σϕ², intercept)
	# turingPrediction=prediction(chain_, xtest_)
	# plot!(xs, turingPrediction, linecolor=:purple, linestyle=:solid, lw=2, label="Turing")

	# studentPrediction=prediction(chain_student, xtest_)
	# plot!(xs, studentPrediction, linecolor=:pink, linestyle=:solid, lw=2, label="T Turing")
end

# ╔═╡ f7235d15-af24-48de-beba-ecb52f55896a
begin
	
	plot(xs, tₜₑₛₜ, ylim= [-0.8, 2.5], linecolor=:black, linestyle=:dash, label="Original")
	scatter!(xsamples, ts, markershape=:xcross, markersize=2.5, markerstrokewidth=0.1, markercolor=:gray, markeralpha=0.9, label="Obvs")
		
	plot!(xs, Φₜₑₛₜ*wₘₗ,  linestyle=:solid, lw=1, label="ML")
		
	
	plot!(xs, Φₜₑₛₜ*wᵥᵢ,  linestyle=:solid, lw=2, label="MFVI")
	plot!(xs, Φₜₑₛₜ*m_fb, linestyle=:solid, lw=2, label="FFVI with fixed hyper")
	plot!(xs, Φₜₑₛₜ*m_fb_, linestyle=:solid, lw=2, label="FFVI with hyper update")
	plot!(xs, Φₜₑₛₜ*m_fb2, linestyle=:solid, lw=2, label="FFVI Laplace prior")

end

# ╔═╡ 4bae7ad5-e93e-452c-b9cf-6098b480999c
begin
	function predictiveLogLik(ws, X, y)
		mc = size(ws)[1]
		h = logistic.(X * ws')
		logLiks = sum(logpdf.(Bernoulli.(h), y), dims=1)
		log(1/mc) + logsumexp(logLiks)
	end
	
	
	function mcPrediction(ws, X)
		# mc = size(ws)[1]
		# h = logistic.(X * ws')
		mean(logistic.(X * ws), dims=2)
	end
	
	function mcPrediction(ws, x1, x2)
		# mc = size(ws)[1]
		# h = logistic.(X * ws')
		mean(logistic.([1.0 x1 x2] * ws))
	end
end

# ╔═╡ de6718b8-b3d1-461b-8d8f-0c575f1208b4
begin
	x1 = range(1, stop=10, length=100)
	x2 = range(1, stop=10, length=100)
	mcVILR = m_lr .+ pack_L(L_lr) * randn(3,100)
	ppfVIfull(x, y) = mcPrediction(mcVILR, x, y)
	contour(x1, x2, ppfVIfull, xlabel="x1", ylabel="x2", fill=true,  connectgaps=true, line_smoothing=0.85, title="Bayesian FFVI prediction", c=:roma)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Distributions = "~0.25.62"
Flux = "~0.13.3"
ForwardDiff = "~0.10.30"
LaTeXStrings = "~1.3.0"
MCMCChains = "~5.3.1"
Plots = "~1.31.1"
StatsFuns = "~1.0.1"
StatsPlots = "~0.14.34"
Turing = "~0.21.6"
Zygote = "~0.6.40"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "5c26c7759412ffcaf0dd6e3172e55d783dd7610b"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.1.3"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Setfield", "SparseArrays"]
git-tree-sha1 = "6320752437e9fbf49639a410017d862ad64415a5"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.5.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "63117898045d6d9e5acbdb517e3808a23aa26436"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.14"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "345effa84030f273ee86fcdd706d8484ce9a1a3c"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.5"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "d7a7dabeaef34e5106cdf6c2ac956e9e3f97f666"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.8"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "9ff1247be1e2aa2e740e84e8c18652bd9d55df22"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.3.8"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "e743af305716a527cdb3a67b31a33a7c3832c41f"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.5"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "1d062b8ab719670c16024105ace35e6d32988d4f"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.18"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5e732808bcf7bbf730e810a9eaafc52705b38bb5"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.13"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "51c842b5a07ad64acdd6cac9e52a304b2d6605b6"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.10.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "e4e5ece72fa2f108fb20c3c5538a5fa9ef3d668a"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.11.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "97fd0a3b7703948a847265156a41079730805c77"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.36.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "332a332c97c7071600984b3c31d9067e1a4e6e25"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "8ccaa8c655bc1b83d2da4d569c9b28254ababd6e"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.2"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0ec161f87bf4ab164ff96dfacf4be8ffff2375fd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.62"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "ec811a2688b3504ce5b315fe7bc86464480d5964"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.41"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "Distributions", "LinearAlgebra", "MacroTools", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "c6f574d855670c2906af3f4053e6db10224e5dda"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.19.3"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterfaceCore", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "4cda4527e990c0cc201286e0a0bfbbce00abcfc2"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "1.0.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "4391d3ed58db9dc5a9883b23a0578316b4798b1f"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.0"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ArrayInterface", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "62350a872545e1369b1d8f11358a21681aa73929"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.3"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "73a4c9447419ce058df716925893e452ba5528ad"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.4.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "4078d3557ab15dd9fe6a0cf6f65e3d4937e98427"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "47f63159f7cb5d0e5e0cfd2f20454adea429bec9"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3a233eeeb2ca45842fe100e0413936834215abf5"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.4+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "57af5939800bce15980bddd2426912c4f83012d8"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.1"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LRUCache]]
git-tree-sha1 = "d64a0aff6691612ab9fb0117b0995270871c5dfc"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.3.0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "dfa6c5f2d5a8918dd97c7f1a9ea0de68c2365426"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.7.5"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8cb9b8fb081afd7728f5de25b9025bff97cb5c7a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.3.1"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "058d08594e91ba1d98dcc3669f9421a76824aa95"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.3"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "b8073fe6973dcfad5fec803dabc1d3a7f6c4ebc8"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.4.3"

[[deps.MLStyle]]
git-tree-sha1 = "2041c1fd6833b3720d363c3ea8140bffaf86d9c4"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.12"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "FLoops", "FoldsThreads", "Random", "ShowCases", "Statistics", "StatsBase", "Transducers"]
git-tree-sha1 = "cf10b2a295df211c6c7e992be73505bf619c1e52"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.8"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "1a80840bcdb73de345230328d49767ab115be6f2"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.8"

[[deps.NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "e161b835c6aa9e2339c1e72c3d4e39891eac7a4f"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.3"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "0e353ed734b1747fc20cd4cba0edd9ac027eff6a"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "ec2e30596282d722f018ae784b7f44f3b88065e4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.6"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9a36165cf84cff35851809a40a928e1103702013"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "afb2b39a354025a6db6decd68f2ef5353e8ff1ae"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.7"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ca433b9e2f5ca3a0ce6702a032fce95a3b6e1e48"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.14"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "93e82cebd5b25eb33068570e3f63a86be16955be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "afeacaecf4ed1649555a19cb2cad3c141bbc9474"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.5.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "ZygoteRules"]
git-tree-sha1 = "7ddd4f1ac52f9cc1b784212785f86a75602a7e4b"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.31.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "30e3981751855e2340e9b524ab58c1ec85c36f33"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArraysCore", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "6a3f7d9b084b508e87d12135de950ac969187954"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.42.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "11f1b69a28b6e4ca1cc18342bfab7adb7ff3a090"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.7.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "9f8a5dc5944dc7fbbe6eb4180660935653b0a9d9"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.0"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6edcea211d224fa551ec8a85debdc6d732f155dc"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "271a7fea12d319f23d55b785c51f6876aadb9ac0"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "642f08bf9ff9e39ccc7b710b2eb9a24971b52b1a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.17"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "43a316e07ae612c461fd874740aeef396c60f5f8"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.34"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "464d64b2510a25e6efe410e7edab14fffdc333df"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.20"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0874c1b5de1b5529b776cfeca3ec0acfada97b1b"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.20"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "DiffResults", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "cba513b222ff87fb55fdccc1a76d26acbc607b0f"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.21.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a49267a2e5f113c7afe93843deea7461c0f6b206"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.40"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═093a22cb-36a4-41d8-9c0e-46cf876192fe
# ╟─51c76828-f55e-11ec-3e26-61ba6bc94c5a
# ╟─f5d9b1cc-63fc-4bdf-9e76-184f8d85cb60
# ╟─06250275-93d3-4004-b2c5-37bc093a6dcb
# ╟─ad647b20-9e7a-4b53-bbc5-7ff0c49c2a3c
# ╟─de714c36-1eac-4a92-8d6c-77947c245894
# ╟─141fccea-1712-4a19-a0ad-e29e83ce7bc6
# ╠═cd2de8de-961a-4365-bac0-3b27b9041159
# ╟─51e61619-585e-4699-99f0-65ad18ec165b
# ╟─0ccdd961-9eb4-4854-90d6-7b41dda103a4
# ╟─ebcca522-2f3f-4d89-bd7a-96f72dba8056
# ╟─41e52510-2d17-4a4d-b6f5-c452d7f84883
# ╟─d29c0514-6a1e-4516-9d27-a0674483e5b5
# ╟─747f77f9-eda4-4a49-8428-88c1b91a68c9
# ╟─f30b5b09-860e-4dc7-91c4-2dff8c6608a3
# ╠═fcb61290-3de5-403a-9183-7668baf6dec6
# ╠═7813bb32-8b29-442d-814b-eb513e50d526
# ╠═f0b3f4d9-2a54-4351-bb73-30320b3fc58e
# ╠═b36d4f31-95ea-4ce6-9409-e3ee1da9c24e
# ╟─60c96652-ffe5-444a-8cda-b5fae6472c77
# ╟─c4351958-ef47-4c69-ba95-da52e7b54fb5
# ╟─f62b5fee-4765-466e-9f26-02a5b119a1e8
# ╟─8ab14983-73fd-4b85-82ad-a0b3404f5918
# ╟─0ed401f5-097c-4206-8336-b751c3b8da17
# ╟─d7646301-f225-4319-839b-855140801f54
# ╟─084f17cb-8301-4708-9c81-c2b8c5f041f7
# ╟─c915353c-3cdc-4949-93a9-3b8b21e3ba4f
# ╟─cfdeee66-e4c4-48fc-af95-287e70fb0ec8
# ╟─8b009cd4-bafc-48aa-9dcd-df92e13d4b6d
# ╠═a1bb4425-17cf-4d62-836f-559020721217
# ╠═3f80ce90-c4d2-4085-80ae-f1ce1b0088a4
# ╠═49f51c64-ffec-47f1-a332-09988442d5ed
# ╟─59154ad8-197c-4ea2-a0a4-fd0e3c862af1
# ╠═fb9b8223-14ec-4e4c-b9a9-f9c44e912008
# ╠═8c2c59b1-063c-4010-af76-de63dec44717
# ╠═c7687b5f-0740-4dd9-b26b-8d29537ddced
# ╠═2d85dce1-d532-42f9-9f1c-33932f3f425f
# ╠═caefb087-9004-4762-8026-c9717ee1187e
# ╠═ef971f6b-844a-44ca-82ec-61cf00f80114
# ╠═75f28936-2b95-4103-a06b-fd4c4908919a
# ╠═7446506a-17fa-4c27-9ef4-dcef0057202a
# ╠═9ceb3db3-da5e-47c8-9244-89c213d8f0b4
# ╠═a3a01e98-c8fc-428d-9264-9759fabe6826
# ╠═ae10fc12-8ddc-4eab-815d-64fd49f5b37d
# ╠═9390258d-37d0-4b42-b80b-676ff6b384df
# ╠═40283cc9-8b14-42c8-8b5c-e3d8114cfd12
# ╠═7bd6e1f7-dbc9-4824-b67b-016d6121bcde
# ╠═03c9a889-108a-4f4e-90e2-29526c3c6ead
# ╟─caafef71-22db-4631-98af-9d4eb39e5acd
# ╠═94b16351-7968-486c-93e9-d80910741cfc
# ╠═34a2a48d-8802-4fb8-bcac-41b3a4de793c
# ╠═7f229bb8-cfc5-4cc1-a603-2244dddad00d
# ╟─ef8a496e-791c-47b5-84ee-af95c6d71df6
# ╠═1b81b6cd-87d1-4c49-bd54-06c57c62182b
# ╟─d79dc084-0cd8-484e-bd17-3d0abce254e1
# ╠═f7235d15-af24-48de-beba-ecb52f55896a
# ╠═ad5fc73a-e2b2-4784-ad74-94a2fb43befe
# ╟─75b2f8a2-228a-4da4-92f7-a4890f53fc54
# ╠═a474cfda-755e-4fed-a6d3-441195c3ed3a
# ╠═0e5ea9e0-c335-4309-92d9-bda23328a230
# ╠═d1b3c121-f35d-491c-834f-6c8de8410df0
# ╠═e78b2b3c-2660-4651-b5d9-4d974af4d451
# ╠═58cd8b62-5f55-4e7a-a5eb-ea372eb78e88
# ╠═de6718b8-b3d1-461b-8d8f-0c575f1208b4
# ╟─0a7e9d02-1a8b-4c9f-a61c-55c5276ecfef
# ╠═15f1b0ab-448a-4bbf-9941-9502623681e5
# ╠═4bae7ad5-e93e-452c-b9cf-6098b480999c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
